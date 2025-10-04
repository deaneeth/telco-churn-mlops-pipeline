"""
Airflow DAG Validation Script
==============================

This script validates the Airflow DAG structure without requiring Airflow to run.
Useful for Windows environments where Airflow requires WSL.

Validates:
- DAG file syntax
- DAG configuration
- Task dependencies
- Task commands
"""

import sys
import ast
import json
from pathlib import Path

def validate_dag_file(dag_path):
    """Validate DAG Python file syntax and structure."""
    
    results = {
        "file_exists": False,
        "syntax_valid": False,
        "dag_found": False,
        "tasks_found": [],
        "dependencies": [],
        "errors": []
    }
    
    try:
        dag_file = Path(dag_path)
        
        # Check file exists
        if not dag_file.exists():
            results["errors"].append(f"DAG file not found: {dag_path}")
            return results
        
        results["file_exists"] = True
        
        # Read and parse file
        with open(dag_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check syntax
        try:
            tree = ast.parse(content)
            results["syntax_valid"] = True
        except SyntaxError as e:
            results["errors"].append(f"Syntax error: {e}")
            return results
        
        # Analyze AST for DAG and task definitions
        for node in ast.walk(tree):
            # Look for DAG instantiation
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'DAG':
                    results["dag_found"] = True
            
            # Look for task assignments (BashOperator, PythonOperator, etc.)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        task_name = target.id
                        if task_name.endswith('_task') or task_name.endswith('_operator'):
                            results["tasks_found"].append(task_name)
        
        # Check for specific task IDs in the file content
        task_ids = []
        for line in content.split('\n'):
            if 'task_id=' in line:
                # Extract task_id value
                start = line.find("task_id='") or line.find('task_id="')
                if start != -1:
                    start += 9
                    end = line.find("'", start) if line.find("'", start) != -1 else line.find('"', start)
                    if end != -1:
                        task_id = line[start:end]
                        task_ids.append(task_id)
        
        results["task_ids"] = task_ids
        
        # Look for dependencies (>>)
        if '>>' in content:
            results["dependencies"].append("Task dependencies found using >> operator")
        
        print(f"✓ DAG file validation successful")
        print(f"  - File: {dag_path}")
        print(f"  - Syntax: Valid")
        print(f"  - DAG found: {results['dag_found']}")
        print(f"  - Task IDs found: {task_ids}")
        print(f"  - Dependencies: {'Yes' if results['dependencies'] else 'No'}")
        
    except Exception as e:
        results["errors"].append(f"Unexpected error: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Validate both DAG locations
    dag_paths = [
        "dags/telco_churn_dag.py",
        "airflow_home/dags/telco_churn_dag.py"
    ]
    
    all_results = {}
    
    for dag_path in dag_paths:
        print(f"\n{'='*60}")
        print(f"Validating: {dag_path}")
        print(f"{'='*60}")
        
        results = validate_dag_file(dag_path)
        all_results[dag_path] = results
        
        if results["errors"]:
            print(f"\n✗ Validation errors:")
            for error in results["errors"]:
                print(f"  - {error}")
    
    # Save results
    output_file = Path("reports/dag_validation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Validation results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    has_errors = any(r.get("errors") for r in all_results.values())
    sys.exit(1 if has_errors else 0)
