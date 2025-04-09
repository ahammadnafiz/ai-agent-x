import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

def list_directory_contents(directory_path: str = ".", include_hidden: bool = False) -> str:
    """Lists all files and directories in the specified directory.
    
    Args:
        directory_path (str): Path to the directory to list. Defaults to current directory.
        include_hidden (bool): Whether to include hidden files (starting with .). Defaults to False.
        
    Returns:
        str: JSON string containing directory contents with file details.
    """
    try:
        # Convert to absolute path for clarity
        abs_path = os.path.abspath(directory_path)
        
        if not os.path.exists(abs_path):
            return json.dumps({"error": f"Directory '{directory_path}' does not exist"})
            
        if not os.path.isdir(abs_path):
            return json.dumps({"error": f"'{directory_path}' is not a directory"})
        
        # Get all contents
        contents = []
        
        for item in os.listdir(abs_path):
            # Skip hidden files if not requested
            if not include_hidden and item.startswith('.'):
                continue
                
            item_path = os.path.join(abs_path, item)
            item_stat = os.stat(item_path)
            
            contents.append({
                "name": item,
                "path": item_path,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size_bytes": item_stat.st_size,
                "last_modified": item_stat.st_mtime,
                "permissions": oct(item_stat.st_mode)[-3:]
            })
        
        return json.dumps({
            "directory": abs_path,
            "item_count": len(contents),
            "contents": contents
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to list directory: {str(e)}"})

def read_file_content(file_path: str, max_size_kb: int = 1024) -> str:
    """Reads and returns the content of a file.
    
    Args:
        file_path (str): Path to the file to read
        max_size_kb (int): Maximum file size to read in KB. Defaults to 1024 (1MB).
        
    Returns:
        str: JSON string containing file content and metadata
    """
    try:
        abs_path = os.path.abspath(file_path)
        
        if not os.path.exists(abs_path):
            return json.dumps({"error": f"File '{file_path}' does not exist"})
            
        if not os.path.isfile(abs_path):
            return json.dumps({"error": f"'{file_path}' is not a file"})
            
        # Check file size
        file_size = os.path.getsize(abs_path) / 1024  # Convert to KB
        if file_size > max_size_kb:
            return json.dumps({
                "error": f"File size ({file_size:.2f} KB) exceeds maximum allowed size ({max_size_kb} KB)"
            })
        
        # Determine if file is binary or text
        try:
            is_binary = False
            with open(abs_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:  # Simple heuristic for binary files
                    is_binary = True
                    
            # Read file content
            if is_binary:
                content = "<binary file content not displayed>"
            else:
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    
            file_stat = os.stat(abs_path)
            
            return json.dumps({
                "file_path": abs_path,
                "size_bytes": file_stat.st_size,
                "last_modified": file_stat.st_mtime,
                "is_binary": is_binary,
                "content": content if not is_binary else None,
                "content_preview": content[:200] + "..." if not is_binary and len(content) > 200 else None
            })
            
        except UnicodeDecodeError:
            # If we hit decode errors, treat as binary
            return json.dumps({
                "file_path": abs_path,
                "size_bytes": os.path.getsize(abs_path),
                "last_modified": os.path.getmtime(abs_path),
                "is_binary": True,
                "content": None,
                "content_preview": "<binary file>"
            })
            
    except Exception as e:
        return json.dumps({"error": f"Failed to read file: {str(e)}"})

def write_file_content(file_path: str, content: str, append: bool = False) -> str:
    """Writes content to a file.
    
    Args:
        file_path (str): Path to the file to write
        content (str): Content to write to the file
        append (bool): Whether to append to the file instead of overwriting. Defaults to False.
        
    Returns:
        str: JSON string containing operation result
    """
    try:
        abs_path = os.path.abspath(file_path)
        
        # Create directories if they don't exist
        directory = os.path.dirname(abs_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Write content
        mode = 'a' if append else 'w'
        with open(abs_path, mode, encoding='utf-8') as f:
            f.write(content)
            
        file_stat = os.stat(abs_path)
        
        return json.dumps({
            "success": True,
            "file_path": abs_path,
            "operation": "append" if append else "write",
            "size_bytes": file_stat.st_size,
            "last_modified": file_stat.st_mtime
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to write to file: {str(e)}"})

def search_files(
    directory_path: str = ".", 
    pattern: str = "*", 
    recursive: bool = True,
    content_match: Optional[str] = None,
    max_results: int = 100
) -> str:
    """Searches for files matching pattern and optionally containing specific content.
    
    Args:
        directory_path (str): Path to directory to search in. Defaults to current directory.
        pattern (str): File pattern to match (glob syntax, e.g., "*.py"). Defaults to "*".
        recursive (bool): Whether to search recursively in subdirectories. Defaults to True.
        content_match (str, optional): Text to search for within files. If None, only matches by name.
        max_results (int): Maximum number of results to return. Defaults to 100.
        
    Returns:
        str: JSON string containing matching files
    """
    try:
        abs_path = os.path.abspath(directory_path)
        
        if not os.path.exists(abs_path):
            return json.dumps({"error": f"Directory '{directory_path}' does not exist"})
            
        if not os.path.isdir(abs_path):
            return json.dumps({"error": f"'{directory_path}' is not a directory"})
            
        # Find all files matching pattern
        matches = []
        count = 0
        
        # Use Path for better glob support
        search_path = Path(abs_path)
        
        # Set glob pattern based on recursivity
        glob_pattern = f"**/{pattern}" if recursive else pattern
        
        for file_path in search_path.glob(glob_pattern):
            if count >= max_results:
                break
                
            if file_path.is_file():
                file_info = {
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "last_modified": file_path.stat().st_mtime
                }
                
                # If content matching is requested
                if content_match is not None:
                    try:
                        # Skip binary files
                        with open(file_path, 'rb') as f:
                            chunk = f.read(1024)
                            if b'\0' in chunk:
                                continue
                                
                        # Search for content match
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if content_match in content:
                                # Find line numbers with matches
                                lines = content.split('\n')
                                matching_lines = [
                                    {"line_num": i+1, "content": line.strip()}
                                    for i, line in enumerate(lines)
                                    if content_match in line
                                ]
                                
                                file_info["content_matches"] = {
                                    "match_count": len(matching_lines),
                                    "matches": matching_lines[:5]  # Limit to 5 matches per file
                                }
                                matches.append(file_info)
                                count += 1
                    except:
                        # Skip files that can't be read
                        continue
                else:
                    # Just name matching
                    matches.append(file_info)
                    count += 1
        
        return json.dumps({
            "search_directory": abs_path,
            "pattern": pattern,
            "recursive": recursive,
            "content_match": content_match,
            "match_count": len(matches),
            "max_results_reached": count >= max_results,
            "matches": matches
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to search files: {str(e)}"})

def file_operations(
    operation: str,
    source_path: str,
    target_path: Optional[str] = None
) -> str:
    """Performs file operations like copy, move, delete, rename.
    
    Args:
        operation (str): Operation to perform: "copy", "move", "delete", or "rename"
        source_path (str): Path to the source file or directory
        target_path (str, optional): Target path for copy, move, or rename operations
        
    Returns:
        str: JSON string containing operation result
    """
    try:
        abs_source = os.path.abspath(source_path)
        
        if not os.path.exists(abs_source):
            return json.dumps({"error": f"Source path '{source_path}' does not exist"})
            
        # Handle different operations
        if operation.lower() == "delete":
            if os.path.isdir(abs_source):
                shutil.rmtree(abs_source)
            else:
                os.remove(abs_source)
                
            return json.dumps({
                "success": True,
                "operation": "delete",
                "source_path": abs_source
            })
            
        elif operation.lower() in ["copy", "move", "rename"]:
            if target_path is None:
                return json.dumps({"error": f"Target path is required for {operation} operation"})
                
            abs_target = os.path.abspath(target_path)
            
            # Create target directory if it doesn't exist
            target_dir = os.path.dirname(abs_target)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            if operation.lower() == "copy":
                if os.path.isdir(abs_source):
                    shutil.copytree(abs_source, abs_target)
                else:
                    shutil.copy2(abs_source, abs_target)
            elif operation.lower() in ["move", "rename"]:
                shutil.move(abs_source, abs_target)
                
            return json.dumps({
                "success": True,
                "operation": operation.lower(),
                "source_path": abs_source,
                "target_path": abs_target
            })
        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})
            
    except Exception as e:
        return json.dumps({"error": f"Failed to perform {operation}: {str(e)}"})