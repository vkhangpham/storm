#!/bin/bash

# Find all report.md files in the results directory
find results/gpt4o/you -name "report.md" | while read report_file; do
    # Get the directory of the report file
    dir=$(dirname "$report_file")
    
    # Construct the path to the corresponding instance_dump.json file
    instance_dump_file="$dir/instance_dump.json"
    
    # Check if the instance_dump.json file exists
    if [ -f "$instance_dump_file" ]; then
        echo "Processing: $report_file"
        # Run the map_citations.py script
        python map_citations.py "$report_file" "$instance_dump_file"
    else
        echo "Warning: No instance_dump.json found for $report_file"
    fi
done

echo "All reports processed."
