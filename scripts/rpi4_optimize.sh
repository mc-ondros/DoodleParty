#!/bin/bash
# RPi4 System Optimization Script for DoodleHunter
# 
# This script configures Raspberry Pi 4 for optimal ML inference performance.
# Run with sudo privileges for system-level optimizations.
#
# Features:
# - CPU governor set to 'performance' mode
# - Thermal monitoring and throttling detection
# - Memory optimization (swap configuration)
# - Process priority configuration
# - System resource monitoring
#
# Usage:
#   sudo bash scripts/rpi4_optimize.sh [--monitor]
#
# Options:
#   --monitor    Start continuous monitoring after optimization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running on RPi4
check_rpi4() {
    echo -e "${BLUE}Checking hardware...${NC}"
    
    if [ ! -f /proc/device-tree/model ]; then
        echo -e "${YELLOW}Warning: Cannot detect hardware model${NC}"
        return
    fi
    
    MODEL=$(cat /proc/device-tree/model)
    echo "Detected: $MODEL"
    
    if [[ ! "$MODEL" =~ "Raspberry Pi 4" ]]; then
        echo -e "${YELLOW}Warning: This script is optimized for Raspberry Pi 4${NC}"
        echo "Current hardware may not benefit from all optimizations."
    else
        echo -e "${GREEN}✓ Raspberry Pi 4 detected${NC}"
    fi
}

# Set CPU governor to performance mode
optimize_cpu() {
    echo -e "\n${BLUE}Optimizing CPU performance...${NC}"
    
    # Check current governor
    CURRENT_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    echo "Current CPU governor: $CURRENT_GOV"
    
    # Set to performance mode
    if [ "$CURRENT_GOV" != "performance" ]; then
        echo "Setting CPU governor to 'performance' mode..."
        for cpu in /sys/devices/system/cpu/cpu[0-3]; do
            if [ -f "$cpu/cpufreq/scaling_governor" ]; then
                echo performance | sudo tee "$cpu/cpufreq/scaling_governor" > /dev/null
            fi
        done
        echo -e "${GREEN}✓ CPU governor set to performance${NC}"
    else
        echo -e "${GREEN}✓ CPU already in performance mode${NC}"
    fi
    
    # Display CPU frequencies
    echo -e "\nCPU Frequencies:"
    for i in 0 1 2 3; do
        FREQ=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq)
        FREQ_MHZ=$((FREQ / 1000))
        echo "  CPU$i: ${FREQ_MHZ} MHz"
    done
}

# Check and display thermal status
check_thermal() {
    echo -e "\n${BLUE}Checking thermal status...${NC}"
    
    # Get CPU temperature
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
        TEMP_C=$((TEMP / 1000))
        
        if [ $TEMP_C -lt 60 ]; then
            echo -e "CPU Temperature: ${GREEN}${TEMP_C}°C${NC} (Good)"
        elif [ $TEMP_C -lt 75 ]; then
            echo -e "CPU Temperature: ${YELLOW}${TEMP_C}°C${NC} (Warm - consider cooling)"
        else
            echo -e "CPU Temperature: ${RED}${TEMP_C}°C${NC} (Hot - throttling likely!)"
        fi
    fi
    
    # Check for throttling
    if command -v vcgencmd &> /dev/null; then
        THROTTLED=$(vcgencmd get_throttled)
        echo "Throttle status: $THROTTLED"
        
        if [[ "$THROTTLED" == *"0x0"* ]]; then
            echo -e "${GREEN}✓ No throttling detected${NC}"
        else
            echo -e "${RED}⚠ Throttling detected! Add cooling or reduce load${NC}"
        fi
    fi
}

# Optimize memory settings
optimize_memory() {
    echo -e "\n${BLUE}Optimizing memory settings...${NC}"
    
    # Check current swap
    SWAP_TOTAL=$(free -m | awk '/Swap:/ {print $2}')
    echo "Current swap: ${SWAP_TOTAL}MB"
    
    # Get swappiness
    SWAPPINESS=$(cat /proc/sys/vm/swappiness)
    echo "Current swappiness: $SWAPPINESS"
    
    # Recommend low swappiness for ML workloads
    if [ $SWAPPINESS -gt 10 ]; then
        echo "Reducing swappiness to 10 (prefer RAM over swap)..."
        echo 10 | sudo tee /proc/sys/vm/swappiness > /dev/null
        echo -e "${GREEN}✓ Swappiness reduced to 10${NC}"
        
        # Make permanent
        if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
            echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf > /dev/null
            echo "  (Made permanent in /etc/sysctl.conf)"
        fi
    else
        echo -e "${GREEN}✓ Swappiness already optimized${NC}"
    fi
    
    # Display memory info
    echo -e "\nMemory Status:"
    free -h | grep -E "Mem:|Swap:"
}

# Check for active cooling
check_cooling() {
    echo -e "\n${BLUE}Checking cooling setup...${NC}"
    
    # This is informational - can't automatically detect fan
    echo "For optimal performance under sustained load:"
    echo "  - Install heatsink on CPU"
    echo "  - Add active cooling (fan)"
    echo "  - Ensure good airflow"
    echo ""
    echo "Target: Keep CPU temperature < 75°C under load"
}

# Display system resources
show_resources() {
    echo -e "\n${BLUE}System Resources:${NC}"
    
    # CPU info
    echo -e "\n${YELLOW}CPU:${NC}"
    lscpu | grep -E "Model name|CPU\(s\)|CPU MHz"
    
    # Memory
    echo -e "\n${YELLOW}Memory:${NC}"
    free -h
    
    # Disk
    echo -e "\n${YELLOW}Disk:${NC}"
    df -h / | tail -1
}

# Monitor system in real-time
monitor_system() {
    echo -e "\n${BLUE}Starting system monitor...${NC}"
    echo "Press Ctrl+C to stop"
    echo ""
    
    while true; do
        clear
        echo -e "${BLUE}=== DoodleHunter RPi4 Monitor ===${NC}"
        echo "$(date)"
        echo ""
        
        # CPU Temperature
        if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
            TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
            TEMP_C=$((TEMP / 1000))
            
            if [ $TEMP_C -lt 60 ]; then
                COLOR=$GREEN
            elif [ $TEMP_C -lt 75 ]; then
                COLOR=$YELLOW
            else
                COLOR=$RED
            fi
            echo -e "CPU Temperature: ${COLOR}${TEMP_C}°C${NC}"
        fi
        
        # CPU Frequencies
        echo -e "\nCPU Frequencies:"
        for i in 0 1 2 3; do
            FREQ=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq 2>/dev/null || echo "0")
            FREQ_MHZ=$((FREQ / 1000))
            echo "  CPU$i: ${FREQ_MHZ} MHz"
        done
        
        # CPU Usage
        echo -e "\nCPU Usage:"
        top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "  Idle: " $1 "%\n  Used: " 100 - $1 "%"}'
        
        # Memory
        echo -e "\nMemory:"
        free -h | grep Mem | awk '{print "  Total: " $2 "\n  Used:  " $3 " (" $3/$2*100 "%)\n  Free:  " $4}'
        
        # Top processes
        echo -e "\nTop Processes (CPU):"
        ps aux --sort=-%cpu | head -6 | tail -5 | awk '{printf "  %-20s %5s%%\n", $11, $3}'
        
        # Throttling check
        if command -v vcgencmd &> /dev/null; then
            THROTTLED=$(vcgencmd get_throttled)
            if [[ "$THROTTLED" != *"0x0"* ]]; then
                echo -e "\n${RED}⚠ THROTTLING DETECTED!${NC}"
            fi
        fi
        
        sleep 2
    done
}

# Set process priority for DoodleHunter
set_priority() {
    echo -e "\n${BLUE}Process Priority Configuration:${NC}"
    echo "To run DoodleHunter with higher priority:"
    echo ""
    echo "  nice -n -10 python src/web/app.py"
    echo ""
    echo "Or with renice for running process:"
    echo "  sudo renice -10 -p \$(pgrep -f 'python.*app.py')"
}

# Main execution
main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   DoodleHunter RPi4 Optimization Script   ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        echo -e "${YELLOW}Warning: Not running as root. Some optimizations may fail.${NC}"
        echo "Run with: sudo bash $0"
        echo ""
    fi
    
    check_rpi4
    optimize_cpu
    check_thermal
    optimize_memory
    check_cooling
    set_priority
    show_resources
    
    echo -e "\n${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Optimization Complete!                 ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    
    # Check for monitor flag
    if [[ "$1" == "--monitor" ]]; then
        monitor_system
    else
        echo -e "\nRun with ${YELLOW}--monitor${NC} flag to start continuous monitoring"
        echo "Example: sudo bash $0 --monitor"
    fi
}

# Run main function
main "$@"
