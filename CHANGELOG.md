# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-06

### Added

- Initial release of RunLocal Hub API client
- Support for CoreML, ONNX, TFLite, and OpenVINO model formats
- Device filtering and selection capabilities
- Benchmark functionality for performance testing on real devices
- Prediction functionality for model inference
- Progress tracking for uploads and downloads with tqdm integration
- Comprehensive error handling with custom exception classes
- Type hints and Pydantic models for data validation
- Rich console output for better user experience
- Asynchronous job submission and polling mechanisms
- Tensor handling for numpy array inputs/outputs
- Authentication via API key
- Support for multiple target devices (Apple, Windows, Android)

### Features

- **Device Management**: Smart filtering by name pattern, SoC type, RAM, year, and compute units
- **Model Support**: Upload and benchmark models in multiple formats
- **Job Polling**: Configurable polling intervals with timeout handling
- **Progress Feedback**: Real-time progress bars for file operations
- **Benchmark Results Visualization**: View multi-device results in a clean table
- **Prediction Outputs Retrieval**: Run inference with uploaded inputs and retrieve outputs for custom validation

### Technical Details

- Requires Python 3.8+
- Built with Pydantic v2 for data validation
- Uses requests for HTTP communication
- Integrated with rich for enhanced console output
- Streaming file uploads for efficient handling of large models
