import pandas as pd
from pathlib import Path

checks_dir = Path('data/kfg/KFG/KFG/checks')

print("=" * 80)
print("KFG SUMMATION CHECK FILES ANALYSIS")
print("=" * 80)
print()

# Analyze each summation type
summation_types = [
    ('pendant_pendant_sum', 'Pendant→Pendant summation'),
    ('indexed_pendant_sum', 'Indexed pendant summation'),
    ('subsidiary_pendant_sum', 'Subsidiary→Pendant summation'),
    ('colored_pendant_sum', 'Colored pendant summation'),
    ('indexed_subsidiary_sum', 'Indexed subsidiary summation'),
    ('group_group_sum', 'Group→Group summation'),
    ('group_sum_bands', 'Group sum bands'),
    ('ascher_decreasing_group', 'Ascher decreasing group'),
    ('pendant_sub_neighbor', 'Pendant-Subsidiary neighbor')
]

print("Summary Statistics:")
print("-" * 80)

total_khipus_with_patterns = set()
total_sum_relationships = 0

for file_prefix, description in summation_types:
    summary_file = checks_dir / f"{file_prefix}.csv"
    relation_file = checks_dir / f"{file_prefix}_relation.csv"
    
    if summary_file.exists():
        df_summary = pd.read_csv(summary_file)
        khipus = len(df_summary)
        
        # Add to total set
        if 'kfg_name' in df_summary.columns:
            total_khipus_with_patterns.update(df_summary['kfg_name'].tolist())
    else:
        khipus = 0
    
    if relation_file.exists():
        df_relations = pd.read_csv(relation_file)
        relations = len(df_relations)
        total_sum_relationships += relations
    else:
        relations = 0
    
    print(f"{description:40} {khipus:4} khipus, {relations:6} relationships")

print()
print(f"Total unique khipus with summation: {len(total_khipus_with_patterns)}")
print(f"Total summation relationships: {total_sum_relationships:,}")
print()

# Detailed look at pendant_pendant_sum
print("Detailed Analysis: Pendant→Pendant Summation")
print("-" * 80)
df = pd.read_csv(checks_dir / 'pendant_pendant_sum.csv')
print(f"Khipus with pendant summation: {len(df)}")
print(f"Total sum cords: {df['num_sum_cords'].sum():,}")
print(f"Avg sum cords per khipu: {df['num_sum_cords'].mean():.1f}")
print()

print("Sample summation patterns (first 5 khipus):")
for i, row in df.head(5).iterrows():
    print(f"  {row['kfg_name']}: {row['num_sum_cords']} sum cords, "
          f"{row['num_left_sums']} left + {row['num_right_sums']} right, "
          f"max_sum_length={row['max_sum_length']}")
print()

# Detailed look at relations
print("Sample Summation Relationships:")
print("-" * 80)
df_rel = pd.read_csv(checks_dir / 'pendant_pendant_sum_relation.csv')
print(f"Total relationships: {len(df_rel):,}")
print()

sample = df_rel.head(3)
for i, row in sample.iterrows():
    print(f"{row['kfg_name']} {row['cord_name']}: value={row['cord_value']}, "
          f"summands={row['num_summands']}")
    print(f"  Formula: {row['summand_string']}")
    if row['has_figure8knot_indicator']:
        print(f"  ✓ Has figure-8 knot indicator")
    print()

print("=" * 80)
print("These files provide GROUND TRUTH for validating our summation detection!")
print("=" * 80)
