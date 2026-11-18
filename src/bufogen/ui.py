"""Compatibility entrypoints for existing studio launchers and ASGI reloaders."""

from bufogen.app.ui import create_app, main

__all__ = ["create_app", "main"]

if __name__ == "__main__":
    main()
