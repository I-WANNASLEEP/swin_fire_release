#!/bin/bash
while true; do
    # ============ 收集所有信息 ============
    
    # 时间
    current_time=$(date)
    
    # CPU 使用率
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # 平均负载
    load_avg=$(uptime | awk -F'load average:' '{print $2}')
    
    # CPU 温度
    cpu_temp_info=""
    if command -v sensors &> /dev/null; then
        temps=$(sensors | grep -oP 'Core \d+:\s+\+\K[0-9.]+' 2>/dev/null)
        if [ -n "$temps" ]; then
            avg_temp=$(echo "$temps" | awk '{sum+=$1; count++} END {printf "%.1f", sum/count}')
            max_temp=$(echo "$temps" | sort -rn | head -1)
            min_temp=$(echo "$temps" | sort -n | head -1)
            cpu_temp_info="CPU 温度: ${avg_temp}°C (最低: ${min_temp}°C, 最高: ${max_temp}°C)"
        else
            other_temp=$(sensors | grep -E "Tctl|Tdie|CPU" | grep -oP '\+\K[0-9.]+' | head -1)
            if [ -n "$other_temp" ]; then
                cpu_temp_info="CPU 温度: ${other_temp}°C"
            else
                cpu_temp_info="CPU 温度: 无法读取"
            fi
        fi
    elif [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        temp_c=$((temp / 1000))
        cpu_temp_info="CPU 温度: ${temp_c}°C"
    else
        cpu_temp_info="CPU 温度: 未检测到传感器"
    fi
    
    # 内存信息 - 直接从 /proc/meminfo 读取
    mem_total_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    mem_available_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    mem_used_kb=$((mem_total_kb - mem_available_kb))
    
    # 转换为人类可读格式
    mem_total_display=$(awk "BEGIN {printf \"%.1fGi\", $mem_total_kb/1024/1024}")
    mem_used_display=$(awk "BEGIN {printf \"%.1fGi\", $mem_used_kb/1024/1024}")
    mem_available_display=$(awk "BEGIN {printf \"%.1fGi\", $mem_available_kb/1024/1024}")
    mem_percent=$(awk "BEGIN {printf \"%.1f\", $mem_used_kb/$mem_total_kb * 100}")
    
    # Swap 信息
    swap_total_kb=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
    swap_free_kb=$(grep SwapFree /proc/meminfo | awk '{print $2}')
    swap_used_kb=$((swap_total_kb - swap_free_kb))
    
    swap_total_display=$(awk "BEGIN {printf \"%.1fGi\", $swap_total_kb/1024/1024}")
    swap_used_display=$(awk "BEGIN {printf \"%.0fMi\", $swap_used_kb/1024}")
    
    # GPU 信息
    gpu_info=""
    if command -v nvidia-smi &> /dev/null; then
        gpu_info="NVIDIA GPU:\n"
        while IFS=',' read -r index name util mem_used mem_total temp power power_limit; do
            gpu_info+="  GPU ${index}: ${name}\n"
            gpu_info+="    使用率: ${util}%\n"
            gpu_info+="    显存: ${mem_used} MB / ${mem_total} MB\n"
            gpu_info+="    温度: ${temp}°C\n"
            if [ "$power_limit" != " [N/A]" ] && [ -n "$power_limit" ]; then
                gpu_info+="    功耗: ${power} W / ${power_limit} W\n"
            else
                gpu_info+="    功耗: ${power} W\n"
            fi
            gpu_info+="\n"
        done < <(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits)
    elif command -v rocm-smi &> /dev/null; then
        gpu_info="AMD GPU:\n"
        gpu_info+=$(rocm-smi --showuse --showmemuse --showtemp)
        gpu_info+="\n"
    else
        gpu_info="未检测到 GPU 或未安装驱动\n"
    fi
    
    # ============ 一次性打印所有信息 ============
    clear
    cat << EOF

${current_time}
┌─────────────────────────────────────┐
│          CPU 使用情况               │
└─────────────────────────────────────┘
CPU 使用率: ${cpu_usage}%
平均负载:${load_avg}
${cpu_temp_info}
┌─────────────────────────────────────┐
│          内存使用情况               │
└─────────────────────────────────────┘
总内存:     ${mem_total_display}
已使用:     ${mem_used_display} (${mem_percent}%)
Swap 总量:  ${swap_total_display}
Swap 使用:  ${swap_used_display}
┌─────────────────────────────────────┐
│          GPU 使用情况               │
└─────────────────────────────────────┘
$(echo -e "${gpu_info}")
EOF
    sleep 2
done

