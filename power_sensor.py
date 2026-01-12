# power_sensor.py
import psutil
import json
import time
from datetime import datetime

OUTPUT_FILE = "power.json"
INTERVAL = 10  # seconds

def get_battery_status():
    battery = psutil.sensors_battery()
    if battery is None:
        return {"battery_percent": None, "charging": None, "status": "No battery detected"}
    
    percent = battery.percent
    charging = battery.power_plugged
    status = "Charging" if charging else "Discharging"
    return {"battery_percent": percent, "charging": charging, "status": status}

def save_to_file(data, filename=OUTPUT_FILE):
    try:
        with open(filename, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), **data}, f, indent=4)
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    while True:
        data = get_battery_status()
        print(f"Battery Status: {data['battery_percent']}% - {data['status']}")
        save_to_file(data)
        time.sleep(INTERVAL)
