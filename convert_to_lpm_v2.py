#!/usr/bin/env python3
"""
Lollms Personality Structure Converter v2
Converts old-style Lollms personalities to the new LPM v2 standard.

Old structure: single config.yaml with all fields
New structure:
  - SOUL.md (contains ALL personality data in markdown)
  - assets/logo.png (icon)
  - scripts/processor.py (optional, for scripted personalities)
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml_safe(filepath: Path) -> Optional[Dict[str, Any]]:
    """Safely load YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def save_text(filepath: Path, content: str) -> bool:
    """Save text content to file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"  Error saving {filepath}: {e}")
        return False


def build_soul_md(old_config: Dict[str, Any]) -> str:
    """
    Build SOUL.md content from old config.
    The SOUL.md contains ALL personality data in markdown format.
    """
    lines = []
    
    # Header with personality name
    name = old_config.get('name', 'Unknown')
    lines.append(f"# {name}")
    lines.append("")
    
    # Description section
    if 'personality_description' in old_config and old_config['personality_description']:
        lines.append("## Description")
        lines.append("")
        lines.append(old_config['personality_description'].strip())
        lines.append("")
    
    # Conditioning section (the core behavior)
    if 'personality_conditioning' in old_config and old_config['personality_conditioning']:
        lines.append("## Conditioning")
        lines.append("")
        lines.append(old_config['personality_conditioning'].strip())
        lines.append("")
    
    # Welcome message section
    if 'welcome_message' in old_config and old_config['welcome_message']:
        lines.append("## Welcome Message")
        lines.append("")
        lines.append(old_config['welcome_message'].strip())
        lines.append("")
    
    # Disclaimer section
    if 'disclaimer' in old_config and old_config['disclaimer']:
        lines.append("## Disclaimer")
        lines.append("")
        lines.append(old_config['disclaimer'].strip())
        lines.append("")
    
    # Metadata section (for reference, not used by LPM v2)
    lines.append("## Metadata")
    lines.append("")
    lines.append("```yaml")
    
    metadata_fields = [
        'name', 'author', 'version', 'category', 'language',
        'dependencies', 'recommended_binding', 'recommended_model',
        'user_message_prefix', 'ai_message_prefix', 'link_text'
    ]
    
    for field in metadata_fields:
        if field in old_config and old_config[field] is not None:
            value = old_config[field]
            if isinstance(value, str):
                lines.append(f"{field}: '{value}'")
            elif isinstance(value, list):
                lines.append(f"{field}: {value}")
            else:
                lines.append(f"{field}: {value}")
    
    # Model parameters
    model_params = {}
    model_fields = [
        ('temperature', 'model_temperature'),
        ('top_k', 'model_top_k'),
        ('top_p', 'model_top_p'),
        ('repeat_penalty', 'model_repeat_penalty'),
        ('repeat_last_n', 'model_repeat_last_n'),
        ('n_predicts', 'model_n_predicts')
    ]
    for new_name, old_name in model_fields:
        if old_name in old_config and old_config[old_name] is not None:
            model_params[new_name] = old_config[old_name]
    
    if model_params:
        lines.append("model_parameters:")
        for key, value in model_params.items():
            lines.append(f"  {key}: {value}")
    
    # Anti prompts
    if 'anti_prompts' in old_config and old_config['anti_prompts']:
        lines.append(f"anti_prompts: {old_config['anti_prompts']}")
    
    lines.append("```")
    lines.append("")
    
    return "\n".join(lines)


def ensure_assets(folder_path: Path) -> Path:
    """Ensure assets folder exists and return its path."""
    assets_path = folder_path / 'assets'
    assets_path.mkdir(parents=True, exist_ok=True)
    return assets_path


def find_logo_file(folder_path: Path) -> Optional[Path]:
    """Find existing logo file in various locations."""
    # Check in assets folder
    assets_path = folder_path / 'assets'
    if assets_path.exists():
        for ext in ['.png', '.jpg', '.jpeg', '.svg', '.gif']:
            logo = assets_path / f'logo{ext}'
            if logo.exists():
                return logo
    return None


def convert_personality(folder_path: Path, dry_run: bool = False, keep_backup: bool = True) -> bool:
    """
    Convert a single personality folder from old to new format.
    """
    config_path = folder_path / 'config.yaml'
    
    if not config_path.exists():
        return False
    
    # Load old config
    old_config = load_yaml_safe(config_path)
    if old_config is None:
        print(f"  Skipped: Could not load config.yaml")
        return False
    
    # Check if already new format (has SOUL.md)
    soul_path = folder_path / 'SOUL.md'
    if soul_path.exists():
        print(f"  Skipped: Already has SOUL.md (new format)")
        return False
    
    # Check if it's actually old format (has personality_conditioning in config.yaml)
    if 'personality_conditioning' not in old_config:
        print(f"  Skipped: No personality_conditioning found (not old format)")
        return False
    
    print(f"\nProcessing: {folder_path}")
    
    if dry_run:
        print(f"  Would convert: {folder_path.name}")
        return True
    
    # Build SOUL.md content
    soul_content = build_soul_md(old_config)
    
    # Ensure assets folder exists
    assets_path = ensure_assets(folder_path)
    
    # Write SOUL.md
    if not save_text(soul_path, soul_content):
        print(f"  ✗ Failed to write SOUL.md")
        return False
    
    # Remove old config.yaml (or backup if requested)
    if keep_backup:
        backup_path = folder_path / 'config.yaml.bak'
        try:
            shutil.move(config_path, backup_path)
            print(f"  Backup created: config.yaml.bak")
        except Exception as e:
            print(f"  Warning: Could not create backup: {e}")
    else:
        try:
            config_path.unlink()
        except Exception as e:
            print(f"  Warning: Could not remove config.yaml: {e}")
    
    print(f"  ✓ Converted successfully")
    print(f"    - SOUL.md (all personality data)")
    
    # Check for logo
    existing_logo = find_logo_file(folder_path)
    if existing_logo:
        print(f"    - assets/{existing_logo.name} (icon exists)")
    else:
        print(f"    - Warning: No logo.png found in assets/")
    
    return True


def find_personality_folders(root_path: Path) -> list[Path]:
    """Find all folders containing old-style config.yaml with personality_conditioning."""
    personality_folders = []
    
    for path in root_path.rglob('config.yaml'):
        folder = path.parent
        # Check if it's old format (has personality_conditioning, no SOUL.md)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'personality_conditioning' in content:
                    # Exclude if already has SOUL.md
                    if not (folder / 'SOUL.md').exists():
                        personality_folders.append(folder)
        except Exception:
            pass
    
    return personality_folders


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert old-style Lollms personalities to LPM v2 standard (SOUL.md format)'
    )
    parser.add_argument(
        'root_path',
        nargs='?',
        default='.',
        help='Root directory to search for personalities (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be converted without making changes'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Require confirmation before each conversion'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not keep config.yaml.bak (delete old config.yaml)'
    )
    
    args = parser.parse_args()
    
    root_path = Path(args.root_path).resolve()
    
    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}")
        sys.exit(1)
    
    print(f"Scanning for old-style Lollms personalities in: {root_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Backup: {'No' if args.no_backup else 'Yes'} (config.yaml.bak)")
    print("-" * 60)
    
    folders = find_personality_folders(root_path)
    
    if not folders:
        print("No old-style personality folders found.")
        sys.exit(0)
    
    print(f"\nFound {len(folders)} old-style personality folder(s) to convert:")
    for f in folders:
        print(f"  - {f.relative_to(root_path)}")
    
    print("-" * 60)
    
    converted = 0
    skipped = 0
    failed = 0
    
    for folder in folders:
        if args.confirm and not args.dry_run:
            response = input(f"\nConvert {folder.name}? [y/N/q]: ").strip().lower()
            if response == 'q':
                print("Aborted by user.")
                break
            if response != 'y':
                print(f"  Skipped by user")
                skipped += 1
                continue
        
        if convert_personality(folder, dry_run=args.dry_run, keep_backup=not args.no_backup):
            converted += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print(f"  Total found:    {len(folders)}")
    print(f"  Converted:      {converted}")
    print(f"  Skipped:        {skipped}")
    print(f"  Failed:         {failed}")
    
    if args.dry_run:
        print("\nThis was a dry run. No files were modified.")
        print("Run without --dry-run to perform actual conversion.")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())