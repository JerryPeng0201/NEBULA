import sys
import json
import numpy as np
from pathlib import Path
import yaml
import warnings

sys.path.append('.')

import os

from nebula.visualization.capability_visualize import plot_radar_comparison
from nebula.visualization.stress_visualize import plot_stress_adaptability

def find_latest_json(directory, json_type):
    """Find the latest JSON file of specified type in directory"""
    results_path = Path(directory)
    if not results_path.exists():
        return None
    
    pattern = f'results_{json_type}_*.json'
    json_files = list(results_path.glob(pattern))
    
    if not json_files:
        return None
    
    # Return the most recently modified file
    return max(json_files, key=lambda f: f.stat().st_mtime)

def load_model_results(json_paths=None, models_dir='../../models'):
    """
    Load results from multiple models
    
    Args:
        json_paths: dict with 'capability' and 'stress' lists of file paths, or 'all'
        models_dir: directory containing model subdirectories
    
    Returns:
        capability_results: list of (model_name, data) tuples
        stress_results: list of (model_name, data) tuples
    """
    capability_results = []
    stress_results = []
    
    # Mode 1: User provides specific JSON paths
    if json_paths and json_paths != 'all':
        for cap_path in json_paths.get('capability', []):
            cap_file = Path(cap_path)
            if cap_file.exists():
                try:
                    with open(cap_file, 'r') as f:
                        data = json.load(f)
                    model_name = data.get('config', {}).get('figure', {}).get('model_name', cap_file.stem)
                    capability_results.append((model_name, data))
                    print(f"✓ Loaded capability: {cap_file}")
                except Exception as e:
                    warnings.warn(f"Failed to load {cap_file}: {e}")
            else:
                warnings.warn(f"Capability file not found: {cap_path}")
        
        for stress_path in json_paths.get('stress', []):
            stress_file = Path(stress_path)
            if stress_file.exists():
                try:
                    with open(stress_file, 'r') as f:
                        data = json.load(f)
                    model_name = data.get('config', {}).get('figure', {}).get('model_name', stress_file.stem)
                    stress_results.append((model_name, data))
                    print(f"✓ Loaded stress: {stress_file}")
                except Exception as e:
                    warnings.warn(f"Failed to load {stress_file}: {e}")
            else:
                warnings.warn(f"Stress file not found: {stress_path}")
    
    # Mode 2: Auto-discover from models directory or use 'all' keyword
    else:
        models_path = Path(models_dir)
        if not models_path.exists():
            warnings.warn(f"Models directory not found: {models_dir}")
            return capability_results, stress_results
        
        # Find all model subdirectories
        for model_dir in models_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            config_file = model_dir / 'config.yaml'
            if not config_file.exists():
                continue
            
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                model_name = config.get('figure', {}).get('model_name', model_dir.name)
                results_dir = model_dir / config['experiment']['save_dir']
                
                # Try to load capability results
                cap_file = find_latest_json(results_dir, 'capability')
                if cap_file:
                    try:
                        with open(cap_file, 'r') as f:
                            data = json.load(f)
                        capability_results.append((model_name, data))
                        print(f"✓ Loaded capability for {model_name}: {cap_file}")
                    except Exception as e:
                        warnings.warn(f"Failed to load capability for {model_name}: {e}")
                else:
                    warnings.warn(f"No capability results found for {model_name}")
                
                # Try to load stress results
                stress_file = find_latest_json(results_dir, 'stress')
                if stress_file:
                    try:
                        with open(stress_file, 'r') as f:
                            data = json.load(f)
                        stress_results.append((model_name, data))
                        print(f"✓ Loaded stress for {model_name}: {stress_file}")
                    except Exception as e:
                        warnings.warn(f"Failed to load stress for {model_name}: {e}")
                else:
                    warnings.warn(f"No stress results found for {model_name}")
                    
            except Exception as e:
                warnings.warn(f"Failed to process {model_dir}: {e}")
    
    return capability_results, stress_results

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

def extract_stress_metrics(all_results):
    """Extract stress test metrics from results"""
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

def generate_graph_multiple(json_paths=None, models_dir='../../models', output_dir='./figures', viz_type='both'):
    """Generate comparison visualizations for multiple models"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading model results...")
    capability_results, stress_results = load_model_results(json_paths, models_dir)
    
    if not capability_results and not stress_results:
        print("No results found to visualize.")
        return
    
    # Generate capability comparison
    if viz_type in ['capability', 'both'] and capability_results:
        print("\nGenerating capability comparison chart...")
        all_scores = []
        model_names = []
        
        for model_name, data in capability_results:
            easy, medium, hard = extract_capability_scores(data)
            all_scores.append((easy, medium, hard))
            model_names.append(model_name)
        
        capability_save_path = output_path / 'capability_comparison.png'
        
        # Use separate mode if only one model, combined for multiple
        mode = 'separate' if len(model_names) == 1 else 'combined'
        
        plot_radar_comparison(
            all_scores=all_scores,
            model_names=model_names,
            save_path=str(capability_save_path),
            labels=["Control", "Perception", "Language", "Spatial", "Dynamic", "Robustness"],
            mode=mode
        )
        print(f"✓ Capability chart saved: {capability_save_path}")
    
    # Generate stress test comparison
    if viz_type in ['stress', 'both'] and stress_results:
        print("Generating stress test comparison chart...")
        
        model_names = []
        all_inference_freq = {'V1': [], 'V2': [], 'V3': []}
        all_latency = {'V1': [], 'V2': [], 'V3': []}
        all_stability = {'V1': [], 'V2': [], 'V3': []}
        all_adaptability = {}
        
        for model_name, data in stress_results:
            model_names.append(model_name)
            
            # Get corresponding capability data if available
            cap_data = next((d for n, d in capability_results if n == model_name), None)
            combined_results = data['results']
            if cap_data:
                combined_results = cap_data['results'] + data['results']
            
            freq, lat, stab, adapt = extract_stress_metrics(combined_results)
            
            for key in ['V1', 'V2', 'V3']:
                all_inference_freq[key].append(freq[key][0])
                all_latency[key].append(lat[key][0])
                all_stability[key].append(stab[key][0])
            
            # Collect adaptability
            for test_type, value in adapt.items():
                if test_type not in all_adaptability:
                    all_adaptability[test_type] = []
                all_adaptability[test_type].append(value[0])
        
        if all_adaptability:
            stress_save_path = output_path / 'stress_comparison.png'
            plot_stress_adaptability(
                models=model_names,
                inference_freq=all_inference_freq,
                latency=all_latency,
                stability=all_stability,
                adaptability=all_adaptability,
                save_path=str(stress_save_path)
            )
            print(f"✓ Stress test chart saved: {stress_save_path}")
    
    print(f"\nVisualization complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comparison visualizations for multiple models')
    parser.add_argument('--capability-jsons', nargs='+', help='Paths to capability JSON files')
    parser.add_argument('--stress-jsons', nargs='+', help='Paths to stress test JSON files')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing model subdirectories with config.yaml')
    parser.add_argument('--output-dir', type=str, default='./models/figures',
                       help='Output directory for figures')
    parser.add_argument('--type', type=str, choices=['capability', 'stress', 'both'],
                       default='both', help='Type of visualization to generate')
    parser.add_argument('--all', action='store_true',
                       help='Auto-discover and visualize all available models')
    args = parser.parse_args()
    
    # Prepare json_paths
    json_paths = None
    if args.capability_jsons or args.stress_jsons:
        json_paths = {
            'capability': args.capability_jsons or [],
            'stress': args.stress_jsons or []
        }
    elif args.all:
        json_paths = 'all'
    
    generate_graph_multiple(
        json_paths=json_paths,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        viz_type=args.type
    )