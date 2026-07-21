import sys
import pexpect
import os

def run_command_with_password(cmd, password, timeout=300):
    print(f"\n🏃 Running local command: {cmd}")
    child = pexpect.spawn(cmd, timeout=timeout)
    # Log stdout to verify progress
    child.logfile = sys.stdout.buffer
    
    while True:
        index = child.expect([
            "Are you sure you want to continue connecting",
            "[Pp]assword:",
            pexpect.EOF,
            pexpect.TIMEOUT
        ])
        if index == 0:
            child.sendline("yes")
        elif index == 1:
            child.sendline(password)
            break
        else:
            print("\n⚠️ Command finished or timed out/failed.")
            return child.before.decode('utf-8', errors='ignore')
            
    child.expect(pexpect.EOF)
    child.close()
    return child.before.decode('utf-8', errors='ignore')

def main():
    ip = "69.62.78.148"
    port = "2222"
    user = "root"
    password = "64X1i7jkQg1@RO--y)L7"
    remote_dir = "/root/agri-doctor"
    
    print("=====================================================================")
    print("🌿 AgriDoctor AI - Automated VPS Deployer (Using deploy_vps.py) 🌿")
    print("=====================================================================")
    print(f"Target VPS: {user}@{ip}:{port}")
    print("=====================================================================")
    
    # Step 1: Create remote directory structure
    mkdir_cmd = f"ssh -o StrictHostKeyChecking=no -p {port} {user}@{ip} 'mkdir -p {remote_dir}/data/uploads/images {remote_dir}/data/uploads/speech'"
    run_command_with_password(mkdir_cmd, password)
    
    # Step 2a: SCP main project files to remote root directory
    main_files = [
        "backend", "config", "docs", "frontend", "src", "tools",
        "Dockerfile", "docker-compose.yml", "nginx.conf", "requirements.txt",
        "requirements-ml.txt", "requirements-serve.txt", "start.sh", ".env", "agridoctor.cloud.conf"
    ]
    valid_main = [f for f in main_files if os.path.exists(f)]
    main_str = " ".join(valid_main)
    
    scp_main_cmd = f"scp -o StrictHostKeyChecking=no -P {port} -r {main_str} {user}@{ip}:{remote_dir}/"
    run_command_with_password(scp_main_cmd, password, timeout=300)

    # Step 2b: SCP specific data files to remote data/ directory
    data_files = [
        "data/taxonomy.json", "data/schemas", "data/instruction_data", "data/models"
    ]
    valid_data = [f for f in data_files if os.path.exists(f)]
    data_str = " ".join(valid_data)
    
    scp_data_cmd = f"scp -o StrictHostKeyChecking=no -P {port} -r {data_str} {user}@{ip}:{remote_dir}/data/"
    run_command_with_password(scp_data_cmd, password, timeout=300)
    
    # Step 3: Run Docker Compose on VPS
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -p {port} {user}@{ip}"
    print(f"\n🏃 Connecting to remote VPS to execute setup and run docker services...")
    child = pexpect.spawn(ssh_cmd, timeout=300)
    child.logfile = sys.stdout.buffer
    
    index = child.expect(["[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT])
    if index == 0:
        child.sendline(password)
    else:
        print("⚠️ Failed to connect via SSH for Docker commands.")
        sys.exit(1)
        
    child.expect([r"#\s*", r"\$\s*"])
    
    # Send remote setup script
    remote_script = """
set -euo pipefail

# Install Docker if not present
if ! [ -x "$(command -v docker)" ]; then
  echo "🐳 Installing Docker..."
  curl -fsSL https://get.docker.com -o get-docker.sh
  sh get-docker.sh
  rm get-docker.sh
fi

# Install Docker Compose if not present
if ! docker compose version &>/dev/null && ! [ -x "$(command -v docker-compose)" ]; then
  echo "🐳 Installing Docker Compose..."
  apt-get update && apt-get install -y docker-compose-plugin || apt-get install -y docker-compose
fi

# Go to project directory
cd /root/agri-doctor
chmod +x start.sh

# Spin up services in background
echo "🚀 Launching containerized application stack..."
if docker compose version &>/dev/null; then
  docker compose down --remove-orphans || true
  docker compose up -d --build
else
  docker-compose down --remove-orphans || true
  docker-compose up -d --build
fi

# Ensure correct permissions for database and model folders
chmod -R 777 /root/agri-doctor/data /root/agri-doctor/models

echo "📊 Container status:"
docker ps
exit
"""
    for line in remote_script.strip().split("\n"):
        child.sendline(line)
        
    child.expect(pexpect.EOF)
    child.close()
    print("\n✅ Deployment script execution finished!")

if __name__ == "__main__":
    main()
