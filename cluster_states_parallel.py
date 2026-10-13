import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Set your output directory here
output_dir = "/home/lmhlanga/repos/malaria_determinants/output/building_metrics/"

def find_metrics_files(output_dir, state_name):
    """Find neighbor and building metrics files for a given state"""
    neighbor_files = glob.glob(os.path.join(output_dir, state_name, "*_neighbor_metrics.csv"))
    building_files = glob.glob(os.path.join(output_dir, state_name, "*_building_metrics.csv"))
    return neighbor_files, building_files

def merge_metrics_data(neighbor_files, building_files):
    """Merge neighbor and building metrics data"""
    all_data = []

    for neighbor_file in neighbor_files:
        try:
            neighbor_data = pd.read_csv(neighbor_file)
            base_name = os.path.basename(neighbor_file).replace('_neighbor_metrics.csv', '')
            
            building_file = next((bf for bf in building_files if base_name in bf), None)

            if building_file:
                building_data = pd.read_csv(building_file)
                if 'Id' in neighbor_data.columns and 'Id' in building_data.columns:
                    merged_data = pd.merge(neighbor_data, building_data, on='Id', how='outer')
                else:
                    merged_data = pd.concat([neighbor_data, building_data], axis=1)
            else:
                merged_data = neighbor_data

            all_data.append(merged_data)
        except Exception as e:
            print(f"Error processing {neighbor_file}: {e}")
            continue

    return pd.concat(all_data, ignore_index=True) if all_data else None

def cluster_building_data(data, state_name, output_dir):
    """Cluster building data using Gaussian Mixture Models"""
    morphology_columns = [col for col in [
        'area_mean', 'nndist_mean', 'perimeter_mean', 'compact_mean',
        'angle_mean', 'shape_mean', 'area', 'perimeter', 'compactness',
        'circularity', 'elongation', 'rectangularity', 'fractal_dimension'
    ] if col in data.columns]

    if len(morphology_columns) < 2:
        morphology_columns = data.select_dtypes(include=[np.number]).columns.tolist()[:6]

    morphology_data = data[morphology_columns].copy().dropna()

    if morphology_data.shape[0] < 10:
        print(f"Not enough data for clustering in {state_name}")
        data['classification'] = np.nan
        return data

    morphology_scaled = StandardScaler().fit_transform(morphology_data)

    try:
        pca = PCA(n_components=0.95)
        morphology_reduced = pca.fit_transform(morphology_scaled)
    except:
        morphology_reduced = morphology_scaled

    def compute_bic(n, X):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            gmm.fit(X)
            return gmm.bic(X)
        except:
            return np.inf

    bic_scores = Parallel(n_jobs=-1)(delayed(compute_bic)(k, morphology_reduced) for k in range(1, 10))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), bic_scores, marker='o')
    plt.title(f'BIC Scores - {state_name}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC')
    plt.grid(True)
    bic_plot_path = os.path.join(output_dir, state_name, f"{state_name}_bic_plot.png")
    plt.savefig(bic_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    best_k = np.argmin(bic_scores) + 1
    final_model = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
    final_model.fit(morphology_reduced)
    labels = final_model.predict(morphology_reduced)

    cluster_data = data.copy()
    cluster_data['classification'] = np.nan
    cluster_data.loc[morphology_data.index, 'classification'] = labels.astype(str)
    return cluster_data

def process_state(state_name, output_dir):
    print(f"\nProcessing state: {state_name}")
    neighbor_files, building_files = find_metrics_files(output_dir, state_name)

    if not neighbor_files and not building_files:
        print(f"No files found for {state_name}")
        return

    merged_data = merge_metrics_data(neighbor_files, building_files)

    if merged_data is None or merged_data.empty:
        print(f"No data to process for {state_name}")
        return

    os.makedirs(os.path.join(output_dir, state_name), exist_ok=True)
    merged_output = os.path.join(output_dir, state_name, f"{state_name}_all_building_metrics.csv")
    merged_data.to_csv(merged_output, index=False)

    clustered_data = cluster_building_data(merged_data, state_name, output_dir)
    clustered_output = os.path.join(output_dir, state_name, f"{state_name}_clustered_building_metrics.csv")
    clustered_data.to_csv(clustered_output, index=False)

    print(f"Finished: {state_name}")

def main():
    state_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    Parallel(n_jobs=-1)(
        delayed(process_state)(state, output_dir) for state in state_dirs
    )

    print("\nAll states processed.")

if __name__ == "__main__":
    main()
