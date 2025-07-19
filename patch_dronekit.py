#!/usr/bin/env python3
"""
Patch for DroneKit to work with newer Python versions and fix common connection issues
"""
import os
import re
import sys
import subprocess
import time

def check_uart_permissions():
    """Check if user has permission to access UART ports"""
    uart_devices = ['/dev/ttyAMA0', '/dev/ttyS0', '/dev/serial0', '/dev/ttyAMA1']
    available_ports = []
    
    print("Checking UART port permissions...")
    for device in uart_devices:
        if os.path.exists(device):
            try:
                # Check if we can open the device for reading
                with open(device, 'rb') as f:
                    available_ports.append(device)
                    print(f"✅ {device} exists and is accessible")
            except PermissionError:
                print(f"❌ {device} exists but permission denied - try 'sudo chmod 666 {device}'")
            except Exception as e:
                print(f"❓ {device} exists but error: {e}")
        else:
            print(f"❌ {device} does not exist")
    
    if not available_ports:
        print("\nNo accessible UART ports found. You may need to:")
        print("1. Enable UART in raspi-config: sudo raspi-config > Interface Options > Serial")
        print("2. Set proper permissions: sudo chmod 666 /dev/ttyAMA0")
        print("3. Add your user to the dialout group: sudo usermod -a -G dialout $USER")
        print("4. Reboot your Raspberry Pi: sudo reboot")
    else:
        print(f"\nAvailable UART ports: {', '.join(available_ports)}")
    
    return available_ports

def check_dronekit_installation():
    """Check if DroneKit is properly installed"""
    try:
        import dronekit
        print(f"DroneKit version: {dronekit.__version__}")
        return True
    except ImportError:
        print("DroneKit is not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "dronekit"])
            print("DroneKit installed successfully!")
            return True
        except Exception as e:
            print(f"Failed to install DroneKit: {e}")
            return False

def patch_dronekit():
    try:
        # Find the dronekit __init__.py file
        import dronekit
        dronekit_path = os.path.abspath(dronekit.__file__)
        print(f"Found DroneKit at: {dronekit_path}")
        
        # Read the file content
        with open(dronekit_path, 'r') as f:
            content = f.read()
        
        # Check if the file needs patching
        if "from collections import OrderedDict, namedtuple, MutableMapping" in content:
            print("Patching DroneKit to use collections.abc.MutableMapping...")
            
            # Replace the import statement
            content = content.replace(
                "from collections import OrderedDict, namedtuple, MutableMapping",
                "from collections import OrderedDict, namedtuple\nfrom collections.abc import MutableMapping"
            )
            
            # Write the patched content back
            with open(dronekit_path, 'w') as f:
                f.write(content)
            
            print("DroneKit successfully patched!")
            return True
        else:
            print("DroneKit seems to be already patched or has a different structure.")
            return True
    except Exception as e:
        print(f"Error patching DroneKit: {e}")
        return False

def patch_connection_timeout():
    """Patch DroneKit connection timeout for more reliable connections"""
    try:
        import dronekit
        dronekit_path = os.path.abspath(dronekit.__file__)
        mavlink_path = os.path.join(os.path.dirname(os.path.dirname(dronekit_path)), 
                                    "pymavlink", "mavutil.py")
        
        if os.path.exists(mavlink_path):
            print(f"Found mavutil.py at: {mavlink_path}")
            with open(mavlink_path, 'r') as f:
                content = f.read()
            
            # Check if DEFAULT_TIMEOUT is too low
            timeout_match = re.search(r'DEFAULT_TIMEOUT\s*=\s*(\d+\.\d+)', content)
            if timeout_match:
                current_timeout = float(timeout_match.group(1))
                print(f"Current DEFAULT_TIMEOUT = {current_timeout}")
                
                if current_timeout < 5.0:
                    print("Increasing DEFAULT_TIMEOUT to 5.0 seconds for better reliability...")
                    content = content.replace(
                        f"DEFAULT_TIMEOUT = {current_timeout}",
                        "DEFAULT_TIMEOUT = 5.0"
                    )
                    
                    with open(mavlink_path, 'w') as f:
                        f.write(content)
                    
                    print("Timeout value patched successfully!")
                    return True
                else:
                    print("DEFAULT_TIMEOUT is already sufficient.")
                    return True
            else:
                print("Could not find DEFAULT_TIMEOUT in mavutil.py")
                return False
        else:
            print(f"Could not find mavutil.py at expected location: {mavlink_path}")
            return False
    except Exception as e:
        print(f"Error patching connection timeout: {e}")
        return False

def test_connection(port='/dev/ttyAMA0', baud=57600, timeout=10):
    """Test connection to the Pixhawk"""
    try:
        from dronekit import connect, APIException
        
        print(f"\nTesting connection to Pixhawk on {port} at {baud} baud...")
        print(f"Connection timeout set to {timeout} seconds")
        print("Attempting connection...")
        
        # Show progress while connecting
        start_time = time.time()
        connection_thread = None
        
        try:
            # Connect with explicit timeout
            vehicle = connect(port, baud=baud, wait_ready=True, heartbeat_timeout=timeout)
            elapsed = time.time() - start_time
            print(f"✅ Successfully connected in {elapsed:.1f} seconds!")
            
            # Display basic info
            print(f" > System status: {vehicle.system_status.state}")
            print(f" > Mode: {vehicle.mode.name}")
            print(f" > GPS: {vehicle.gps_0.fix_type}")
            print(f" > Battery: {vehicle.battery.voltage}V")
            
            # Clean up
            vehicle.close()
            return True
        except APIException as e:
            elapsed = time.time() - start_time
            print(f"❌ Connection timed out after {elapsed:.1f} seconds: {e}")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    except ImportError:
        print("❌ DroneKit not available. Run the script first to install and patch it.")
        return False

if __name__ == "__main__":
    print("=== DroneKit Diagnostic and Patching Tool ===\n")
    
    # First check DroneKit installation
    dronekit_ok = check_dronekit_installation()
    
    if dronekit_ok:
        # Apply patches
        patch_dronekit()
        patch_connection_timeout()
        
        # Check UART port permissions
        available_ports = check_uart_permissions()
        
        # Test connection if ports are available
        if available_ports:
            print("\nWould you like to test the Pixhawk connection? (y/n)")
            choice = input("> ").lower()
            if choice == 'y':
                for port in available_ports:
                    if test_connection(port):
                        print(f"\n✅ Connection test succeeded on {port}!")
                        print(f"Use this port in your application: {port}")
                        break
                    else:
                        print(f"\n❌ Connection test failed on {port}, trying next port if available...")
    
    print("\nDiagnostic complete. Restart your application if any patches were applied.")
