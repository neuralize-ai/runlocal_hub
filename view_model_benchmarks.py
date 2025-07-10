#!/usr/bin/env python3

from runlocal_hub import RunLocalClient, display_benchmark_results


def main():
    client = RunLocalClient()

    model_ids = client.get_models_ids()

    benchmark_data = client.get_model_benchmarks(model_ids[0])

    display_benchmark_results(benchmark_data, show_ram_usage=True)


if __name__ == "__main__":
    main()
