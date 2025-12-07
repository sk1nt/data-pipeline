#!/usr/bin/env python3
"""CLI for managing allowlist for automated alerts.

Usage:
  manage_allowlist.py list users
  manage_allowlist.py list channels
  manage_allowlist.py add user 12345
  manage_allowlist.py add channel 67890
  manage_allowlist.py remove user 12345
  manage_allowlist.py remove channel 67890
"""
import argparse
import sys

from services.auth_service import AuthService


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage allowlist for automated alerts")
    subparsers = parser.add_subparsers(dest="action")

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("type", choices=["users", "channels"], help="Type to list")

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("type", choices=["user", "channel"])
    add_parser.add_argument("id", help="ID to add")

    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("type", choices=["user", "channel"])
    remove_parser.add_argument("id", help="ID to remove")

    args = parser.parse_args(argv)
    if args.action == "list":
        if args.type == "users":
            users = AuthService.list_users_allowlist()
            for u in users:
                print(u)
        else:
            channels = AuthService.list_channels_allowlist()
            for c in channels:
                print(c)
    elif args.action == "add":
        if args.type == "user":
            if AuthService.add_user_to_allowlist(args.id):
                print(f"Added user {args.id}")
            else:
                print(f"Failed to add user {args.id}")
        else:
            if AuthService.add_channel_to_allowlist(args.id):
                print(f"Added channel {args.id}")
            else:
                print(f"Failed to add channel {args.id}")
    elif args.action == "remove":
        if args.type == "user":
            if AuthService.remove_user_from_allowlist(args.id):
                print(f"Removed user {args.id}")
            else:
                print(f"Failed to remove user {args.id}")
        else:
            if AuthService.remove_channel_from_allowlist(args.id):
                print(f"Removed channel {args.id}")
            else:
                print(f"Failed to remove channel {args.id}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
