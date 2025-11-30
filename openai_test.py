import os
import sys
import subprocess

def main():
    """
    Top-level execution script for the RCA Agent.
    Runs the agent logic located in src/rca_agent.py without importing internal packages directly.
    """
    # Get the absolute path of the current directory (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the agent script
    agent_script = os.path.join(project_root, "src", "rca_agent.py")
    
    # Check if the script exists
    if not os.path.exists(agent_script):
        print(f"Error: Agent script not found at {agent_script}")
        sys.exit(1)
        
    print(f"Starting RCA analysis...")
    print(f"Project Root: {project_root}")
    
    # Prepare the environment
    env = os.environ.copy()
    # Ensure PYTHONPATH includes the project root if needed
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        # Execute the agent script as a subprocess
        # This avoids importing 'src' modules in this script, preventing path/import errors
        subprocess.run([sys.executable, agent_script], cwd=project_root, env=env, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"\nRCA analysis failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nRCA analysis interrupted by user.")
        sys.exit(130)

if __name__ == "__main__":
    main()
