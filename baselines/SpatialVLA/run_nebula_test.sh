#!/bin/bash

CONFIG_FILE="config.yaml"

echo "Starting NEBULA SpatialVLA evaluation experiments..."
echo "================================================"

# Run capability tests
echo "Running capability tests..."
python spatialvla_simulation.py --config $CONFIG_FILE --test-type capability
CAPABILITY_EXIT_CODE=$?

if [ $CAPABILITY_EXIT_CODE -ne 0 ]; then
    echo "Error: Capability tests failed with exit code $CAPABILITY_EXIT_CODE"
    exit $CAPABILITY_EXIT_CODE
fi

# Run stress tests
echo "Running stress tests..."
python spatialvla_simulation.py --config $CONFIG_FILE --test-type stress
STRESS_EXIT_CODE=$?

if [ $STRESS_EXIT_CODE -ne 0 ]; then
    echo "Error: Stress tests failed with exit code $STRESS_EXIT_CODE"
    exit $STRESS_EXIT_CODE
fi

echo "================================================"
echo "All experiments completed."

# Generate visualizations
echo "Generating visualization figures..."
python -c "
import yaml
import sys
sys.path.append('../../')
from nebula.visualization.generate_graph_single import visualize_results

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

figure_config = config.get('figure', {})
if figure_config.get('save_capability_figure') or figure_config.get('save_stress_figure'):
    viz_type = 'both'
    if figure_config.get('save_capability_figure') and not figure_config.get('save_stress_figure'):
        viz_type = 'capability'
    elif figure_config.get('save_stress_figure') and not figure_config.get('save_capability_figure'):
        viz_type = 'stress'
    
    visualize_results(
        model_name=figure_config.get('model_name', 'Model'),
        results_dir=config['experiment']['save_dir'],
        output_dir=figure_config.get('save_dir', './figures'),
        viz_type=viz_type
    )
    print('Visualization complete.')
else:
    print('Figure generation disabled in config.')
"

VISUALIZATION_EXIT_CODE=$?

if [ $VISUALIZATION_EXIT_CODE -ne 0 ]; then
    echo "Warning: Visualization generation failed with exit code $VISUALIZATION_EXIT_CODE"
    echo "Experiment results are saved, but figures were not generated."
else
    echo "Results and figures saved successfully."
fi