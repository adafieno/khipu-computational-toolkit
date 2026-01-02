"""
Khipu Function Classifier

Classify khipus as "Accounting" vs "Narrative/Administrative" based on:
1. Numeric content coverage
2. Summation patterns
3. Color diversity
4. Structural complexity

Based on cluster analysis:
- Cluster 6 (9.3% numeric) = likely non-accounting prototype
- Clusters 0-5 (66.8-77% numeric) = likely accounting

References:
- Urton 2017: Khipus may encode both numeric and narrative information
- Medrano & Khosla 2024: Summation patterns indicate accounting function
"""

import pandas as pd
import numpy as np
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Tuple
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore


class KhipuFunctionClassifier:
    """Classify khipus by functional type."""

    def __init__(self, db_path: Path, config):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load all relevant features for classification."""
        print("Loading data...")

        # Load features
        features = pd.read_csv(
            self.config.get_processed_file(
                'graph_structural_features.csv', phase=4))
        clusters = pd.read_csv(
            self.config.get_processed_file(
                'cluster_assignments_kmeans.csv', phase=4))
        summation = pd.read_csv(
            self.config.get_processed_file(
                'summation_test_results.csv', phase=3))

        # Load color diversity (unique colors per khipu)
        color_data = pd.read_csv(
            self.config.get_processed_file(
                'color_data.csv', phase=2))
        color_diversity = color_data.groupby(
            'khipu_id')['color_cd_1'].nunique().reset_index()
        color_diversity.columns = ['khipu_id', 'color_diversity']

        # Merge all data
        data = features[['khipu_id', 'num_nodes', 'depth', 'avg_branching', 'has_numeric']].merge(
            clusters[['khipu_id', 'cluster']], on='khipu_id'
        ).merge(
            summation[['khipu_id', 'has_pendant_summation', 'pendant_match_rate']],
            on='khipu_id', how='left'
        ).merge(
            color_diversity, on='khipu_id', how='left'
        )

        # Fill missing values
        data['has_pendant_summation'] = data['has_pendant_summation'].fillna(
            False)
        data['pendant_match_rate'] = data['pendant_match_rate'].fillna(0.0)
        data['color_diversity'] = data['color_diversity'].fillna(0)

        # Calculate numeric coverage (percentage of cords with numeric values)
        data['numeric_coverage'] = data['has_numeric']

        print(f"✓ Loaded {len(data)} khipus")
        return data

    def create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create training labels based on cluster characteristics."""
        # Cluster 6 = Non-Accounting (9.3% numeric)
        # Other clusters = Accounting (66.8-77% numeric)
        data['function_label'] = data['cluster'].apply(
            lambda x: 'Non-Accounting' if x == 6 else 'Accounting'
        )

        accounting_count = (data['function_label'] == 'Accounting').sum()
        non_accounting_count = (
            data['function_label'] == 'Non-Accounting').sum()

        print("\nTraining labels:")
        print(f"  Accounting: {accounting_count} khipus")
        print(f"  Non-Accounting: {non_accounting_count} khipus")

        return data

    def extract_features(
            self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and labels."""
        feature_cols = [
            'numeric_coverage',
            'has_pendant_summation',
            'pendant_match_rate',
            'color_diversity',
            'num_nodes',
            'depth',
            'avg_branching'
        ]

        X = data[feature_cols].values
        y = (data['function_label'] == 'Accounting').astype(int).values

        return X, y

    def train_classifier(
            self,
            X: np.ndarray,
            y: np.ndarray) -> RandomForestClassifier:
        """Train random forest classifier."""
        print("\nTraining classifier...")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train random forest
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(X_scaled, y)

        print("✓ Classifier trained")

        return clf, scaler

    def classify_all_khipus(
            self,
            data: pd.DataFrame,
            clf,
            scaler) -> pd.DataFrame:
        """Classify all khipus and compute confidence scores."""
        print("\nClassifying all khipus...")

        feature_cols = [
            'numeric_coverage',
            'has_pendant_summation',
            'pendant_match_rate',
            'color_diversity',
            'num_nodes',
            'depth',
            'avg_branching'
        ]

        X = data[feature_cols].values
        X_scaled = scaler.transform(X)

        # Predict probabilities
        proba = clf.predict_proba(X_scaled)
        predictions = clf.predict(X_scaled)

        # Add results to dataframe
        data['predicted_accounting'] = predictions
        data['accounting_probability'] = proba[:, 1]
        data['predicted_function'] = data['predicted_accounting'].apply(
            lambda x: 'Accounting' if x == 1 else 'Non-Accounting'
        )

        accounting_count = data['predicted_accounting'].sum()
        non_accounting_count = len(data) - accounting_count

        print("Classification complete:")
        print(
            f"  Accounting: {accounting_count} khipus ({accounting_count/len(data)*100:.1f}%)")
        print(
            f"  Non-Accounting: {non_accounting_count} khipus ({non_accounting_count/len(data)*100:.1f}%)")

        return data

    def analyze_feature_importance(self, clf) -> Dict:
        """Analyze which features are most important for classification."""
        feature_names = [
            'Numeric Coverage',
            'Has Summation',
            'Summation Accuracy',
            'Color Diversity',
            'Khipu Size',
            'Hierarchy Depth',
            'Branching Factor'
        ]

        importance = clf.feature_importances_
        importance_dict = dict(zip(feature_names, importance))

        print("\nFeature Importance:")
        for name, imp in sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.3f}")

        return importance_dict

    def validate_against_provenance(self, data: pd.DataFrame) -> Dict:
        """Validate classifications against geographic patterns."""
        print("\nValidating against provenance patterns...")

        # Load provenance
        provenance = pd.read_sql_query(
            "SELECT KHIPU_ID, PROVENANCE FROM khipu_main",
            self.conn
        )

        merged = data.merge(
            provenance,
            left_on='khipu_id',
            right_on='KHIPU_ID',
            how='left')
        merged['PROVENANCE'] = merged['PROVENANCE'].fillna('Unknown')

        # Filter to major provenances
        major_provs = merged['PROVENANCE'].value_counts().head(5).index
        validation_data = merged[merged['PROVENANCE'].isin(major_provs)]

        # Calculate accounting rate by provenance
        prov_stats = []
        for prov in major_provs:
            prov_data = validation_data[validation_data['PROVENANCE'] == prov]
            if len(prov_data) > 0:
                accounting_rate = prov_data['predicted_accounting'].mean(
                ) * 100
                avg_numeric = prov_data['numeric_coverage'].mean() * 100
                avg_summation = prov_data['has_pendant_summation'].mean() * 100

                prov_stats.append({
                    'provenance': prov,
                    'count': len(prov_data),
                    'accounting_rate': accounting_rate,
                    'avg_numeric_coverage': avg_numeric,
                    'avg_summation_rate': avg_summation
                })

                print(f"\n{prov} (n={len(prov_data)}):")
                print(f"  Accounting: {accounting_rate:.1f}%")
                print(f"  Avg numeric coverage: {avg_numeric:.1f}%")
                print(f"  Summation rate: {avg_summation:.1f}%")

        return {'provenance_stats': prov_stats}

    def export_results(
            self,
            data: pd.DataFrame,
            importance: Dict,
            validation: Dict,
            output_dir: Path):
        """Export classification results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export classifications
        output_cols = [
            'khipu_id',
            'cluster',
            'predicted_function',
            'accounting_probability',
            'numeric_coverage',
            'has_pendant_summation',
            'pendant_match_rate',
            'color_diversity',
            'num_nodes',
            'depth',
            'avg_branching']

        output_path = output_dir / "khipu_function_classification.csv"
        data[output_cols].to_csv(output_path, index=False)
        print(f"\n✓ Exported classifications to {output_path}")

        # Export analysis summary
        summary = {
            'total_khipus': len(data),
            'accounting_count': int(
                data['predicted_accounting'].sum()),
            'non_accounting_count': int(
                (data['predicted_accounting'] == 0).sum()),
            'feature_importance': importance,
            'provenance_validation': validation}

        summary_path = output_dir / "function_classification_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Exported summary to {summary_path}")


def main():
    print("=" * 80)
    print("KHIPU FUNCTION CLASSIFICATION")
    print("=" * 80)
    print()

    # Get configuration
    config = get_config()

    # Validate setup
    validation = config.validate_setup()
    if not validation['valid']:
        print("\nConfiguration errors:")
        for error in validation['errors']:
            print(f"  • {error}")
        sys.exit(1)

    print(f"Database: {config.get_database_path()}")
    print()

    classifier = KhipuFunctionClassifier(config.get_database_path(), config)

    # Load data
    data = classifier.load_data()

    # Create training labels based on clusters
    data = classifier.create_labels(data)

    # Extract features and train
    X, y = classifier.extract_features(data)
    clf, scaler = classifier.train_classifier(X, y)

    # Analyze feature importance
    importance = classifier.analyze_feature_importance(clf)

    # Classify all khipus
    data = classifier.classify_all_khipus(data, clf, scaler)

    # Validate against provenance
    validation_results = classifier.validate_against_provenance(data)

    # Export results
    output_dir = config.processed_dir / 'phase8'
    classifier.export_results(data, importance, validation_results, output_dir)

    print()
    print("=" * 80)
    print("CLASSIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
