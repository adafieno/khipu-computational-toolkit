"""
Sequence Prediction for Confidence Improvement

Predicts improved values for low-confidence cords (confidence < 0.5) based on:
1. Summation constraints (parent = sum of children)
2. Sibling patterns (similar values in adjacent cords)
3. Position-based patterns

Three approaches:
1. Constraint-based inference (using summation)
2. Statistical prediction (mean/median of siblings)
3. ML-based prediction (Random Forest on context features)

Compares predicted values against current low-confidence values to assess
potential improvements in accuracy and confidence scores.

Usage: python scripts/predict_missing_values.py
"""

import sys
from pathlib import Path

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore
import json  # noqa: E402
import networkx as nx  # noqa: E402
from sklearn.model_selection import cross_val_score, GroupKFold  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def load_data():
    """
    Load cord data and graph structures.

    Uses centralized configuration for file paths.
    Normalizes column names to uppercase for consistency.
    """
    print("Loading khipu data...")

    # Get configuration
    config = get_config()

    # Validate setup
    validation = config.validate_setup()
    if not validation['valid']:
        print("\nConfiguration errors:")
        for error in validation['errors']:
            print(f"  • {error}")
        sys.exit(1)

    # Load cord hierarchy from Phase 2 and numeric values with confidence from Phase 1
    hierarchy_path = config.get_processed_file('cord_hierarchy.csv', phase=2)
    cord_values_path = config.get_processed_file('cord_numeric_values.csv', phase=1)

    print(f"  Loading hierarchy from: {hierarchy_path}")
    print(f"  Loading values from: {cord_values_path}")

    hierarchy = pd.read_csv(hierarchy_path)
    cord_values = pd.read_csv(cord_values_path)

    # Normalize column names to uppercase for consistency
    hierarchy.columns = hierarchy.columns.str.upper()
    cord_values.columns = cord_values.columns.str.upper()

    # Drop confidence from hierarchy to avoid conflict (values confidence is what we want)
    if 'CONFIDENCE' in hierarchy.columns:
        hierarchy = hierarchy.drop(columns=['CONFIDENCE'])

    # Merge with hierarchy
    data = hierarchy.merge(
        cord_values[['CORD_ID', 'NUMERIC_VALUE', 'CONFIDENCE']],
        on='CORD_ID',
        how='left',
        suffixes=('', '_KNOT')
    )

    # Build graphs from hierarchy data
    # NOTE: We use PENDANT_FROM as the canonical parent field because it explicitly
    # represents the pendant-from relationship. ATTACHED_TO is used in some contexts
    # but PENDANT_FROM is more reliable for hierarchical structure.
    graphs = {}
    for khipu_id in data['KHIPU_ID'].unique():
        G = nx.DiGraph()
        khipu_data = data[data['KHIPU_ID'] == khipu_id]

        # Add all cords as nodes
        for _, row in khipu_data.iterrows():
            G.add_node(row['CORD_ID'], level=row.get('CORD_LEVEL', 0))

        # Add edges from parent to child using PENDANT_FROM
        for _, row in khipu_data.iterrows():
            parent_id = row['PENDANT_FROM']
            if pd.notna(parent_id) and parent_id != 0 and parent_id in G:
                G.add_edge(parent_id, row['CORD_ID'])

        graphs[khipu_id] = G

    print(f"Loaded {len(data)} cords from {data['KHIPU_ID'].nunique()} khipus")
    print(f"Built {len(graphs)} graphs")
    print(f"Average confidence: {data['CONFIDENCE'].mean():.3f}")
    print(f"Low-confidence cords (<0.5): {(data['CONFIDENCE'] < 0.5).sum()} ({(data['CONFIDENCE'] < 0.5).mean()*100:.1f}%)")
    print(f"High-confidence cords (>=0.5): {(data['CONFIDENCE'] >= 0.5).sum()} ({(data['CONFIDENCE'] >= 0.5).mean()*100:.1f}%)")

    return data, graphs


def constraint_based_prediction(data, graphs):
    """
    Predict improved values for low-confidence cords using summation constraints.

    If a parent has a value and all but one child have values,
    we can infer an improved child value using summation.

    Targets cords with confidence < 0.5 for potential improvement.

    Improved validation:
    - Predicted value must be >= 0
    - Sum of siblings must not exceed parent (with small tolerance)
    - Confidence based on parent value magnitude
    """
    print(f"\n{'='*60}")
    print("CONSTRAINT-BASED PREDICTION")
    print(f"{'='*60}\n")

    predictions = []
    TOLERANCE_RATIO = 0.05  # Allow 5% tolerance for measurement error
    CONFIDENCE_THRESHOLD = 0.5

    # Get low-confidence cords
    low_conf_data = data[data['CONFIDENCE'] < CONFIDENCE_THRESHOLD].copy()
    print(f"Targeting {len(low_conf_data)} low-confidence cords (confidence < {CONFIDENCE_THRESHOLD})")

    for khipu_id in data['KHIPU_ID'].unique():
        if khipu_id not in graphs:
            continue

        G = graphs[khipu_id]
        khipu_data = data[data['KHIPU_ID'] == khipu_id].copy()
        low_conf_cords = set(low_conf_data[low_conf_data['KHIPU_ID'] == khipu_id]['CORD_ID'])

        # Create cord_id -> value and confidence mappings
        value_map = dict(zip(khipu_data['CORD_ID'], khipu_data['NUMERIC_VALUE']))
        confidence_map = dict(zip(khipu_data['CORD_ID'], khipu_data['CONFIDENCE']))

        # Check each low-confidence node
        for node in low_conf_cords:
            if node not in G.nodes():
                continue

            # Get parent
            parents = list(G.predecessors(node))
            if not parents:
                continue
            parent = parents[0]

            # Get parent value
            parent_value = value_map.get(parent)
            if pd.isna(parent_value):
                continue

            # Get siblings (other children of same parent)
            siblings = list(G.successors(parent))
            sibling_values = [value_map.get(s) for s in siblings if s != node]

            # Check if all siblings have values
            if all(pd.notna(v) for v in sibling_values):
                # Infer missing value from summation constraint
                predicted_value = parent_value - sum(sibling_values)

                # Validation: predicted value should be reasonable
                tolerance = parent_value * TOLERANCE_RATIO

                # Accept if:
                # 1. Non-negative (with small tolerance for measurement error)
                # 2. Sum doesn't exceed parent by more than tolerance
                if predicted_value >= -tolerance and sum(sibling_values) <= parent_value + tolerance:
                    # Clamp negative values to 0
                    predicted_value = max(0, predicted_value)

                    # Confidence based on parent value magnitude
                    new_confidence = 0.95 if parent_value > 10 else 0.85
                    current_value = value_map.get(node, 0)
                    current_confidence = confidence_map.get(node, 0)

                    predictions.append({
                        'khipu_id': khipu_id,
                        'cord_id': node,
                        'method': 'constraint_summation',
                        'predicted_value': predicted_value,
                        'current_value': current_value,
                        'value_change': abs(predicted_value - current_value),
                        'current_confidence': current_confidence,
                        'predicted_confidence': new_confidence,
                        'confidence_gain': new_confidence - current_confidence,
                        'parent_value': parent_value,
                        'num_siblings': len(siblings) - 1
                    })

    pred_df = pd.DataFrame(predictions)
    print(f"Predictions made: {len(pred_df)}")

    if len(pred_df) > 0:
        print("\nValue statistics:")
        print(f"  Mean predicted value: {pred_df['predicted_value'].mean():.2f}")
        print(f"  Mean value change: {pred_df['value_change'].mean():.2f}")
        print(f"  Mean confidence gain: {pred_df['confidence_gain'].mean():.3f}")

        print("\nTop 10 predictions by confidence gain:")
        print("-" * 80)
        for _, row in pred_df.nlargest(10, 'confidence_gain').iterrows():
            print(f"Khipu {row['khipu_id']} | Cord {row['cord_id']} | "
                  f"Current: {row['current_value']:.0f} ({row['current_confidence']:.2f}) → "
                  f"Predicted: {row['predicted_value']:.0f} ({row['predicted_confidence']:.2f}) | "
                  f"Gain: +{row['confidence_gain']:.3f}")

    return pred_df


def sibling_based_prediction(data, graphs):
    """
    Predict improved values for low-confidence cords based on sibling patterns.

    Uses median of sibling values as prediction.
    Targets cords with confidence < 0.5 for potential improvement.
    """
    print(f"\n{'='*60}")
    print("SIBLING-BASED PREDICTION")
    print(f"{'='*60}\n")

    predictions = []
    CONFIDENCE_THRESHOLD = 0.5

    # Get low-confidence cords
    low_conf_data = data[data['CONFIDENCE'] < CONFIDENCE_THRESHOLD].copy()
    print(f"Targeting {len(low_conf_data)} low-confidence cords (confidence < {CONFIDENCE_THRESHOLD})")

    for khipu_id in data['KHIPU_ID'].unique():
        if khipu_id not in graphs:
            continue

        G = graphs[khipu_id]
        khipu_data = data[data['KHIPU_ID'] == khipu_id].copy()
        low_conf_cords = set(low_conf_data[low_conf_data['KHIPU_ID'] == khipu_id]['CORD_ID'])

        # Create cord_id -> value and confidence mappings
        value_map = dict(zip(khipu_data['CORD_ID'], khipu_data['NUMERIC_VALUE']))
        confidence_map = dict(zip(khipu_data['CORD_ID'], khipu_data['CONFIDENCE']))

        # Check each low-confidence node
        for node in low_conf_cords:
            if node not in G.nodes():
                continue

            # Get parent
            parents = list(G.predecessors(node))
            if not parents:
                continue
            parent = parents[0]

            # Get siblings
            siblings = [s for s in G.successors(parent) if s != node]
            sibling_values = [value_map.get(s) for s in siblings if pd.notna(value_map.get(s))]

            # Need at least 2 siblings with high-confidence values
            high_conf_siblings = [v for v, conf in zip(sibling_values, 
                                  [confidence_map.get(s, 0) for s in siblings if pd.notna(value_map.get(s))])
                                  if conf >= 0.7]
            
            if len(high_conf_siblings) >= 2:
                # Use median of high-confidence sibling values
                predicted_value = np.median(high_conf_siblings)
                sibling_std = np.std(high_conf_siblings) if len(high_conf_siblings) > 1 else 0

                # Confidence based on sibling agreement (lower std = higher confidence)
                new_confidence = 0.75 if sibling_std < 10 else 0.65
                current_value = value_map.get(node, 0)
                current_confidence = confidence_map.get(node, 0)

                predictions.append({
                    'khipu_id': khipu_id,
                    'cord_id': node,
                    'method': 'sibling_median',
                    'predicted_value': predicted_value,
                    'current_value': current_value,
                    'value_change': abs(predicted_value - current_value),
                    'current_confidence': current_confidence,
                    'predicted_confidence': new_confidence,
                    'confidence_gain': new_confidence - current_confidence,
                    'num_siblings_with_values': len(high_conf_siblings),
                    'sibling_mean': np.mean(high_conf_siblings),
                    'sibling_std': sibling_std
                })

    pred_df = pd.DataFrame(predictions)
    print(f"Predictions made: {len(pred_df)}")

    if len(pred_df) > 0:
        print("\nValue statistics:")
        print(f"  Mean predicted value: {pred_df['predicted_value'].mean():.2f}")
        print(f"  Mean value change: {pred_df['value_change'].mean():.2f}")
        print(f"  Mean confidence gain: {pred_df['confidence_gain'].mean():.3f}")
        print(f"  Mean sibling agreement (std): {pred_df['sibling_std'].mean():.2f}")

        print("\nTop 10 predictions by confidence gain:")
        print("-" * 80)
        for _, row in pred_df.nlargest(10, 'confidence_gain').iterrows():
            print(f"Khipu {row['khipu_id']} | Cord {row['cord_id']} | "
                  f"Current: {row['current_value']:.0f} ({row['current_confidence']:.2f}) → "
                  f"Predicted: {row['predicted_value']:.0f} ({row['predicted_confidence']:.2f}) | "
                  f"Gain: +{row['confidence_gain']:.3f}")

    return pred_df


def ml_based_prediction(data, graphs):
    """
    Predict improved values for low-confidence cords using ML on context features.

    Features:
    - Cord level in hierarchy
    - Number of siblings
    - Parent value
    - Position among siblings
    - Khipu-level statistics

    Trains on high-confidence cords (>= 0.7) and predicts for low-confidence cords (< 0.5).
    """
    print(f"\n{'='*60}")
    print("ML-BASED PREDICTION (Random Forest)")
    print(f"{'='*60}\n")

    # Build feature matrix for all cords
    features_list = []

    for khipu_id in data['KHIPU_ID'].unique():
        if khipu_id not in graphs:
            continue

        G = graphs[khipu_id]
        khipu_data = data[data['KHIPU_ID'] == khipu_id].copy()

        # Khipu-level stats
        khipu_mean = khipu_data['NUMERIC_VALUE'].mean()
        khipu_median = khipu_data['NUMERIC_VALUE'].median()

        # Create cord_id -> value mapping
        value_map = dict(zip(khipu_data['CORD_ID'], khipu_data['NUMERIC_VALUE']))

        for node in G.nodes():
            # Get parent
            parents = list(G.predecessors(node))
            parent = parents[0] if parents else None
            parent_value = value_map.get(parent) if parent else np.nan

            # Get siblings
            if parent:
                siblings = list(G.successors(parent))
                num_siblings = len(siblings) - 1
                sibling_position = siblings.index(node) if node in siblings else 0
            else:
                num_siblings = 0
                sibling_position = 0

            # Get level
            try:
                level = len(nx.shortest_path(G, list(G.nodes())[0], node)) - 1
            except (nx.NetworkXNoPath, nx.NodeNotFound, IndexError):
                level = 0

            # Get children
            children = list(G.successors(node))
            num_children = len(children)

            # Get confidence for this cord
            cord_confidence = value_map.get(node)
            cord_row = khipu_data[khipu_data['CORD_ID'] == node]
            if len(cord_row) > 0:
                cord_confidence_score = cord_row['CONFIDENCE'].values[0]
            else:
                cord_confidence_score = 0

            features_list.append({
                'khipu_id': khipu_id,
                'cord_id': node,
                'level': level,
                'num_siblings': num_siblings,
                'sibling_position': sibling_position,
                'parent_value': parent_value,
                'num_children': num_children,
                'khipu_mean': khipu_mean,
                'khipu_median': khipu_median,
                'target_value': value_map.get(node),
                'confidence': cord_confidence_score
            })

    features_df = pd.DataFrame(features_list)

    # Split into training (high confidence >= 0.7) and prediction (low confidence < 0.5)
    train_df = features_df[features_df['confidence'] >= 0.7].copy()
    predict_df = features_df[features_df['confidence'] < 0.5].copy()

    print(f"Training samples (confidence >= 0.7): {len(train_df)}")
    print(f"Prediction samples (confidence < 0.5): {len(predict_df)}")

    if len(train_df) < 100:
        print("Not enough training data for ML prediction")
        return pd.DataFrame()

    # Prepare features
    feature_cols = ['level', 'num_siblings', 'sibling_position', 'parent_value',
                    'num_children', 'khipu_mean', 'khipu_median']

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target_value']

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Cross-validation with GroupKFold to prevent data leakage across khipus
    # (cords from same khipu should not be split across train/test)
    print("\nPerforming grouped cross-validation (by khipu)...")
    groups = train_df['khipu_id']
    group_kfold = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        rf, X_train, y_train,
        groups=groups,
        cv=group_kfold,
        scoring='neg_mean_absolute_error'
    )
    print(f"Cross-validation MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    print("(Note: Grouped by khipu to prevent leakage)")

    # Feature importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature importances:")
    for _, row in importances.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.3f}")

    # Make predictions
    if len(predict_df) > 0:
        X_predict = predict_df[feature_cols].fillna(0)
        predictions = rf.predict(X_predict)

        predict_df['predicted_value'] = predictions
        predict_df['method'] = 'random_forest'
        
        # Calculate improvements
        predict_df['current_value'] = predict_df['target_value']
        predict_df['value_change'] = abs(predict_df['predicted_value'] - predict_df['current_value'])
        predict_df['current_confidence'] = predict_df['confidence']
        predict_df['predicted_confidence'] = 0.70  # ML predictions get moderate confidence
        predict_df['confidence_gain'] = predict_df['predicted_confidence'] - predict_df['current_confidence']

        # Filter out unreasonable predictions
        predict_df = predict_df[predict_df['predicted_value'] >= 0]

        print(f"\nPredictions made: {len(predict_df)}")
        print("Value statistics:")
        print(f"  Mean predicted value: {predict_df['predicted_value'].mean():.2f}")
        print(f"  Mean value change: {predict_df['value_change'].mean():.2f}")
        print(f"  Mean confidence gain: {predict_df['confidence_gain'].mean():.3f}")

        print("\nTop 10 predictions by confidence gain:")
        print("-" * 80)
        for _, row in predict_df.nlargest(10, 'confidence_gain').iterrows():
            print(f"Khipu {row['khipu_id']} | Cord {row['cord_id']} | "
                  f"Current: {row['current_value']:.0f} ({row['current_confidence']:.2f}) → "
                  f"Predicted: {row['predicted_value']:.0f} ({row['predicted_confidence']:.2f}) | "
                  f"Gain: +{row['confidence_gain']:.3f}")

        return predict_df[['khipu_id', 'cord_id', 'method', 'predicted_value', 'current_value', 
                           'value_change', 'current_confidence', 'predicted_confidence', 'confidence_gain']]

    return pd.DataFrame()


def combine_predictions(constraint_pred, sibling_pred, ml_pred):
    """Combine predictions from all methods with priority ordering."""
    print(f"\n{'='*60}")
    print("COMBINING PREDICTIONS")
    print(f"{'='*60}\n")

    # Priority: Constraint > Sibling > ML
    all_preds = []

    if len(constraint_pred) > 0:
        all_preds.append(constraint_pred)
    if len(sibling_pred) > 0:
        all_preds.append(sibling_pred)
    if len(ml_pred) > 0:
        all_preds.append(ml_pred)

    if not all_preds:
        print("No predictions made by any method")
        return pd.DataFrame()

    combined = pd.concat(all_preds, ignore_index=True)

    # Keep best prediction per cord (prioritize by method)
    method_priority = {'constraint_summation': 1, 'sibling_median': 2, 'random_forest': 3}
    combined['priority'] = combined['method'].map(method_priority)
    combined = combined.sort_values(['cord_id', 'priority']).drop_duplicates('cord_id', keep='first')

    print(f"Total unique predictions: {len(combined)}")
    print("\nBy method:")
    for method in combined['method'].unique():
        count = (combined['method'] == method).sum()
        avg_gain = combined[combined['method'] == method]['confidence_gain'].mean()
        print(f"  {method:25s}: {count:4d} ({count/len(combined)*100:5.1f}%) | Avg gain: +{avg_gain:.3f}")

    print("\nOverall improvement metrics:")
    print(f"  Mean confidence gain: +{combined['confidence_gain'].mean():.3f}")
    print(f"  Mean value change: {combined['value_change'].mean():.2f}")
    print(f"  Cords with gain > 0.3: {(combined['confidence_gain'] > 0.3).sum()} ({(combined['confidence_gain'] > 0.3).mean()*100:.1f}%)")

    return combined


def save_results(combined, constraint_pred, sibling_pred, ml_pred):
    """Save prediction results."""
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}\n")

    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    # Save combined predictions
    if len(combined) > 0:
        output_file = output_dir / "cord_value_predictions.csv"
        combined.to_csv(output_file, index=False)
        print(f"✓ Saved predictions: {output_file} ({len(combined)} cords)")

    # Save method-specific results
    if len(constraint_pred) > 0:
        constraint_file = output_dir / "constraint_based_predictions.csv"
        constraint_pred.to_csv(constraint_file, index=False)
        print(f"✓ Saved constraint predictions: {constraint_file}")

    if len(sibling_pred) > 0:
        sibling_file = output_dir / "sibling_based_predictions.csv"
        sibling_pred.to_csv(sibling_file, index=False)
        print(f"✓ Saved sibling predictions: {sibling_file}")

    if len(ml_pred) > 0:
        ml_file = output_dir / "ml_based_predictions.csv"
        ml_pred.to_csv(ml_file, index=False)
        print(f"✓ Saved ML predictions: {ml_file}")

    # Save summary
    summary = {
        'total_predictions': len(combined) if len(combined) > 0 else 0,
        'by_method': combined['method'].value_counts().to_dict() if len(combined) > 0 else {},
        'improvement_metrics': {
            'mean_confidence_gain': float(combined['confidence_gain'].mean()) if len(combined) > 0 else 0,
            'mean_value_change': float(combined['value_change'].mean()) if len(combined) > 0 else 0,
            'high_gain_count': int((combined['confidence_gain'] > 0.3).sum()) if len(combined) > 0 else 0
        },
        'method_counts': {
            'constraint': len(constraint_pred),
            'sibling': len(sibling_pred),
            'ml': len(ml_pred)
        }
    }

    summary_file = output_dir / "value_prediction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_file}")


def main():
    """Main prediction pipeline."""
    print(f"\n{'='*70}")
    print(" CORD VALUE PREDICTION FOR CONFIDENCE IMPROVEMENT ")
    print(f"{'='*70}\n")

    # Load data
    data, graphs = load_data()

    # Run three prediction methods
    constraint_pred = constraint_based_prediction(data, graphs)
    sibling_pred = sibling_based_prediction(data, graphs)
    ml_pred = ml_based_prediction(data, graphs)

    # Combine predictions
    combined = combine_predictions(constraint_pred, sibling_pred, ml_pred)

    # Save results
    save_results(combined, constraint_pred, sibling_pred, ml_pred)

    print(f"\n{'='*70}")
    print(" CONFIDENCE IMPROVEMENT ANALYSIS COMPLETE ")
    print(f"{'='*70}\n")

    print("Review the following files:")
    print("  • data/processed/cord_value_predictions.csv - Combined predictions with improvements")
    print("  • data/processed/constraint_based_predictions.csv - Summation-based improvements")
    print("  • data/processed/sibling_based_predictions.csv - Sibling pattern-based improvements")
    print("  • data/processed/ml_based_predictions.csv - Random Forest predictions")
    print("  • data/processed/value_prediction_summary.json - Summary statistics")


if __name__ == "__main__":
    main()
