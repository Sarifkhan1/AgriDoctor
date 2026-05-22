import sys
import pexpect

def ssh_run(command):
    ip = "69.62.78.148"
    port = "2222"
    user = "root"
    password = "64X1i7jkQg1@RO--y)L7"
    
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -p {port} {user}@{ip}"
    
    child = pexpect.spawn(ssh_cmd, timeout=30)
    
    index = child.expect([
        "password:",
        "Password:",
        pexpect.EOF,
        pexpect.TIMEOUT
    ])
    
    if index == 0 or index == 1:
        child.sendline(password)
        child.expect([r"#\s*", r"\$\s*", pexpect.EOF, pexpect.TIMEOUT])
    else:
        return f"Failed to reach password prompt: {child.before.decode('utf-8', errors='ignore')}"
        
    child.sendline(command)
    child.expect([r"#\s*", r"\$\s*", pexpect.EOF, pexpect.TIMEOUT])
    output = child.before.decode('utf-8', errors='ignore')
    
    lines = output.splitlines()
    if lines:
        output = "\n".join(lines[1:])
        
    child.sendline("exit")
    child.close()
    
    return output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ssh_run.py <command>")
        sys.exit(1)
        
    cmd = " ".join(sys.argv[1:])
    result = ssh_run(cmd)
    print(result)
