def parse_metadata_block(text: str) -> tuple[str, dict]:
    """
    Parse a metadata block and return the source URL and a dictionary of metadata fields.
    
    Args:
        text (str): Text block containing metadata information
        
    Returns:
        tuple: (source_url, metadata_dict) where metadata_dict contains field types as keys 
        and lists of (value, description) tuples as values
    """
    lines = text.split('\n')
    metadata = {}
    source_url = ""
    current_field = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and separator lines
        if not line or line.startswith('==='):
            continue
            
        # Extract source URL
        if line.startswith('Source URL:'):
            source_url = line.replace('Source URL:', '').strip()
            continue
            
        # Check for field type
        if line and not line.startswith('-'):
            if line.endswith(':'):
                current_field = line[:-1]  # Remove trailing colon
                metadata[current_field] = []
            continue
            
        # Process metadata entries
        if line.startswith('- (') and current_field:
            # Extract content between parentheses
            content = line[5:-1]  # Remove "  - (" and ")"
            if ',' in content:
                # Split into value and description
                value, desc = content.split(',', 1)
                value = value.strip()
                desc = desc.strip()
                metadata[current_field].append((value, desc))
    
    return source_url, metadata

def read_metadata_file(file_path: str) -> list[tuple[str, dict]]:
    """
    Read the metadata file and parse all blocks.
    
    Args:
        file_path (str): Path to the metadata file
        
    Returns:
        list: List of (source_url, metadata_dict) tuples for each block
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split content into blocks using the separator
    sep_blocks = content.split('\n==================================================\n')
    blocks = []

    print(len(sep_blocks))

    if len(sep_blocks)%2:
        for i in range(1, len(sep_blocks)-1, 2):
            block = sep_blocks[i] + "\n" + sep_blocks[i+1]
            blocks.append(block)
    else:
        raise ValueError("The file content does not have the expected format.")

    
    # Parse each non-empty block
    results = []
    for block in blocks:
        if block.strip():
            url, metadata = parse_metadata_block(block)
            if url:  # Only add blocks with a valid source URL
                results.append((url, metadata))
    
    return results

# Example usage:
file_path = "outputs/2025-05-20entity_type_map.txt"
metadata_blocks = read_metadata_file(file_path)

# Print example of first block
if metadata_blocks:
    url, metadata = metadata_blocks[0]
    print(f"Source URL: {url}")
    print("\nExample fields:")
    for field, values in list(metadata.items())[:3]:  # Show first 3 fields
        print(f"\n{field}:")
        for value, desc in values:
            print(f"  - {value}: {desc}")