"""
Display NEBULA version information.

Usage:
    python -m nebula.scripts.version
    python -m nebula.scripts.version --full
    python -m nebula.scripts.version --check-deps
"""

import argparse
import sys


def check_dependencies():
    """Check if required dependencies meet version requirements."""
    from nebula._version import __requires__
    
    print("\n" + "=" * 60)
    print("Dependency Version Check")
    print("=" * 60)
    
    all_ok = True
    
    # Check Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    required_python = __requires__.get("python", "N/A")
    print(f"Python:     {python_version} (required: {required_python})")
    
    # Check other dependencies
    deps_to_check = [
        ("sapien", "sapien"),
        ("torch", "torch"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
    ]
    
    for display_name, module_name in deps_to_check:
        required = __requires__.get(display_name, "N/A")
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            status = "✓"
        except ImportError:
            version = "NOT INSTALLED"
            status = "✗"
            all_ok = False
        
        print(f"{display_name:12s} {version:15s} (required: {required}) {status}")
    
    print("=" * 60)
    
    if all_ok:
        print("✓ All dependencies are installed")
    else:
        print("✗ Some dependencies are missing")
    
    return all_ok


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Display NEBULA version information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Display full version information including features and dependencies",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check installed dependency versions",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Display only the version number",
    )
    
    parsed_args = parser.parse_args(args)
    
    from nebula._version import (
        __version__,
        __git_commit__,
        __git_branch__,
        __git_tag__,
        __build_date__,
        get_full_version_info,
    )
    
    if parsed_args.short:
        # Just print the version number
        print(__version__)
    elif parsed_args.full:
        # Print everything
        print(get_full_version_info())
        if parsed_args.check_deps:
            check_dependencies()
    elif parsed_args.check_deps:
        # Version + dependency check
        print(f"\nNEBULA version: {__version__}")
        check_dependencies()
    else:
        # Default: basic version info
        print(f"\nNEBULA Version Information")
        print("=" * 60)
        print(f"Version:        {__version__}")
        print(f"Git Commit:     {__git_commit__}")
        print(f"Git Branch:     {__git_branch__}")
        if __git_tag__:
            print(f"Git Tag:        {__git_tag__}")
        print(f"Build Date:     {__build_date__}")
        print("=" * 60)
        print("\nUse --full to see complete information")
        print("Use --check-deps to verify dependencies")


if __name__ == "__main__":
    main()

