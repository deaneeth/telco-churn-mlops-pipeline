"""
Simple DAG syntax validation without running Airflow
"""
import ast
import sys

def validate_dag_syntax(dag_file):
    """Validate Python syntax and extract DAG information"""
    try:
        with open(dag_file, 'r') as f:
            code = f.read()
        
        # Parse the Python file
        tree = ast.parse(code)
        
        print(f"✓ Syntax validation passed for {dag_file}")
        
        # Extract some basic information
        dag_id = None
        tasks = []
        
        for node in ast.walk(tree):
            # Look for DAG instantiation
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'id') and node.func.id == 'DAG':
                    for keyword in node.keywords:
                        if keyword.arg == 'dag_id':
                            if isinstance(keyword.value, ast.Constant):
                                dag_id = keyword.value.value
            
            # Look for task assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if hasattr(target, 'id') and target.id.endswith('_task'):
                        tasks.append(target.id)
        
        print(f"\nDAG Information:")
        print(f"  DAG ID: {dag_id}")
        print(f"  Tasks found: {', '.join(tasks)}")
        print(f"  Total tasks: {len(tasks)}")
        
        # Check for bash commands in the file
        bash_commands = []
        for line in code.split('\n'):
            if 'bash_command=' in line:
                bash_commands.append(line.strip())
        
        print(f"\nBash Commands:")
        for cmd in bash_commands:
            print(f"  {cmd}")
        
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error in {dag_file}:")
        print(f"  Line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"✗ Error validating {dag_file}: {e}")
        return False

if __name__ == '__main__':
    dag_file = 'dags/telco_churn_dag.py'
    success = validate_dag_syntax(dag_file)
    sys.exit(0 if success else 1)
