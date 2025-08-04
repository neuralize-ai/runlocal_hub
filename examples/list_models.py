#!/usr/bin/env python3

from runlocal_hub import RunLocalClient, display_model


def main():
    client = RunLocalClient()

    models = client.get_models()

    if not models:
        print("No models found for your account.")
        return

    print(f"Found {len(models)} model(s):\n")

    for i, model in enumerate(models, 1):
        print(f"Model {i}:")
        display_model(model)
        print()


if __name__ == "__main__":
    main()