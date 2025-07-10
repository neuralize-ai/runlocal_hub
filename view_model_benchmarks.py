#!/usr/bin/env python3

from runlocal_hub import RunLocalClient, display_benchmark_results, display_model


def main():
    client = RunLocalClient()

    model_id = client.get_models_ids()[0]

    model_info = client.get_model(model_id)

    display_model(model_info)

    benchmark_data = client.get_model_benchmarks(model_id)

    display_benchmark_results(benchmark_data, show_ram_usage=True)


if __name__ == "__main__":
    main()
