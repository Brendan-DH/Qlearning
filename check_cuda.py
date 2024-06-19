from torch.cuda import is_available, get_device_name, get_device_properties, device_count

print("Cuda is available: " + ("YES" if is_available() else "NO"))

count = device_count()
print(f"{count} devices found")
for i in range(0, count): 
    print(f"Device {i}")
    print("\tDevice name: " + get_device_name(i))
    props = get_device_properties(i)
    print("\tDevice properties: ") 
    print(f"\t\tCUDA Device Name: {props.name}")
    print(f"\t\tCompute Capability: major={props.major} / minor={props.minor}")
    print(f"\t\tTotal Memory: {props.total_memory}")
    print(f"\t\tMultiprocessors: {props.multi_processor_count}")
