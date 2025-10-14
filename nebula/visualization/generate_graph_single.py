import sys
import json
import numpy as np
from pathlib import Path

# add nebula to sys.path
sys.path.append('../../')

from nebula.visualization.capability_visualize import plot_radar_comparison
from nebula.visualization.stress_visualize import plot_stress_adaptability

def load_json_files(results_dir):
    """Load capability and stress test JSON files from directory"""
    results_path = Path(results_dir)
    
    capability_file = None
    stress_file = None

    json_list = list(results_path.glob('*.json'))

    capability_files = [json_file for json_file in json_list if 'results_capability' in json_file.name.lower()]
    stress_files = [json_file for json_file in json_list if 'results_stress' in json_file.name.lower()]

    # load the latest file if multiple exist
    if capability_files:
        capability_file = max(capability_files, key=lambda f: f.stat().st_mtime)
    if stress_files:
        stress_file = max(stress_files, key=lambda f: f.stat().st_mtime)
    
    if not capability_file:
        raise FileNotFoundError(f"No capability results file found in {results_dir}")
    if not stress_file:
        raise FileNotFoundError(f"No stress results file found in {results_dir}")
    
    print(f"Loading capability results from: {capability_file}")
    print(f"Loading stress results from: {stress_file}")
    
    with open(capability_file, 'r') as f:
        capability_data = json.load(f)
    
    with open(stress_file, 'r') as f:
        stress_data = json.load(f)
    
    return capability_data, stress_data

def extract_capability_scores(results):
    """Extract capability test scores organized by difficulty level"""
    categories = ['Control', 'Perception', 'Language', 'Spatial', 'Dynamic', 'Robust']
    
    easy_scores = {cat: [] for cat in categories}
    medium_scores = {cat: [] for cat in categories}
    hard_scores = {cat: [] for cat in categories}
    
    for task_result in results['results']:
        task_name = task_result['task']
        success_rate = task_result['success_rate']
        
        parts = task_name.split('-')
        if len(parts) < 3:
            continue
            
        category = parts[0]
        difficulty = parts[-1]
        
        if category not in categories:
            continue
        
        if difficulty == 'Easy':
            easy_scores[category].append(success_rate)
        elif difficulty == 'Medium':
            medium_scores[category].append(success_rate)
        elif difficulty == 'Hard':
            hard_scores[category].append(success_rate)
    
    easy_avg = [np.mean(easy_scores[cat]) if easy_scores[cat] else 0 for cat in categories]
    medium_avg = [np.mean(medium_scores[cat]) if medium_scores[cat] else 0 for cat in categories]
    hard_avg = [np.mean(hard_scores[cat]) if hard_scores[cat] else 0 for cat in categories]
    
    return easy_avg, medium_avg, hard_avg

def extract_stress_metrics(capability_data, stress_data):
    """Extract stress test metrics from both capability and stress results"""
    
    all_results = capability_data['results'] + stress_data['results']
    
    easy_metrics = {'freq': [], 'latency': [], 'stability': []}
    medium_metrics = {'freq': [], 'latency': [], 'stability': []}
    hard_metrics = {'freq': [], 'latency': [], 'stability': []}
    
    adaptability_tasks = {}
    
    for task_result in all_results:
        task_name = task_result['task']
        
        if task_name.startswith('AdaptationTest'):
            test_type = task_name.replace('AdaptationTest-', '').replace('_', ' ')
            adaptability_tasks[test_type] = task_result['success_rate']
            continue
        
        avg_freq = task_result.get('avg_inference_frequency_hz', 0)
        avg_latency = task_result.get('avg_latency_ms', 0)
        avg_stability = task_result.get('avg_stability_score', 0)
        
        if 'Easy' in task_name:
            easy_metrics['freq'].append(avg_freq)
            easy_metrics['latency'].append(avg_latency)
            easy_metrics['stability'].append(avg_stability)
        elif 'Medium' in task_name:
            medium_metrics['freq'].append(avg_freq)
            medium_metrics['latency'].append(avg_latency)
            medium_metrics['stability'].append(avg_stability)
        elif 'Hard' in task_name:
            hard_metrics['freq'].append(avg_freq)
            hard_metrics['latency'].append(avg_latency)
            hard_metrics['stability'].append(avg_stability)
    
    inference_freq = {
        'V1': [np.mean(easy_metrics['freq'])] if easy_metrics['freq'] else [0],
        'V2': [np.mean(medium_metrics['freq'])] if medium_metrics['freq'] else [0],
        'V3': [np.mean(hard_metrics['freq'])] if hard_metrics['freq'] else [0]
    }
    
    latency = {
        'V1': [np.mean(easy_metrics['latency'])] if easy_metrics['latency'] else [0],
        'V2': [np.mean(medium_metrics['latency'])] if medium_metrics['latency'] else [0],
        'V3': [np.mean(hard_metrics['latency'])] if hard_metrics['latency'] else [0]
    }
    
    stability = {
        'V1': [np.mean(easy_metrics['stability'])] if easy_metrics['stability'] else [0],
        'V2': [np.mean(medium_metrics['stability'])] if medium_metrics['stability'] else [0],
        'V3': [np.mean(hard_metrics['stability'])] if hard_metrics['stability'] else [0]
    }
    
    adaptability = {k: [v] for k, v in adaptability_tasks.items()}
    
    return inference_freq, latency, stability, adaptability

def visualize_results(model_name, results_dir, output_dir='./figures', viz_type='both'):
    """Generate visualization plots from evaluation results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    capability_data, stress_data = load_json_files(results_dir)
    
    cap_session = capability_data['session_id']
    stress_session = stress_data['session_id']
    
    # Generate capability radar chart
    if viz_type in ['capability', 'both']:
        print("\nGenerating capability radar chart...")
        easy_scores, medium_scores, hard_scores = extract_capability_scores(capability_data)
        
        all_scores = [(easy_scores, medium_scores, hard_scores)]
        model_names = [model_name]
        
        capability_save_path = output_path / f'capability_radar_{cap_session}.png'
        plot_radar_comparison(
            all_scores=all_scores,
            model_names=model_names,
            save_path=str(capability_save_path),
            labels=["Control", "Perception", "Language", "Spatial", "Dynamic", "Robustness"],
            mode="separate"
        )
        print(f"  - Capability chart: {capability_save_path}")
    
    # Generate stress test plots
    if viz_type in ['stress', 'both']:
        print("Generating stress test charts...")
        inference_freq, latency, stability, adaptability = extract_stress_metrics(
            capability_data, stress_data
        )
        
        if adaptability:
            stress_save_path = output_path / f'stress_test_{stress_session}.png'
            plot_stress_adaptability(
                models=[model_name],
                inference_freq=inference_freq,
                latency=latency,
                stability=stability,
                adaptability=adaptability,
                save_path=str(stress_save_path)
            )
            print(f"  - Stress test chart: {stress_save_path}")
    
    print(f"\nVisualization complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='Model', 
                       help='Name of the model (for labeling)')
    parser.add_argument('--results-dir', type=str, required=True, 
                       help='Directory containing results JSON files')
    parser.add_argument('--output-dir', type=str, default='./figures', 
                       help='Output directory for figures')
    parser.add_argument('--type', type=str, choices=['capability', 'stress', 'both'], 
                       default='both', help='Type of visualization to generate')
    args = parser.parse_args()
    
    visualize_results(args.model_name, args.results_dir, args.output_dir, args.type)