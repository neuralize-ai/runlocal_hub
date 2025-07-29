import json
from typing import Any, Dict, Iterator, Optional, Union

import requests

from ..exceptions import APIError, AuthenticationError
from ..utils.decorators import with_retry


class HTTPClient:
    """
    HTTP client for making API requests with authentication and error handling.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        debug: bool = False,
    ):
        """
        Initialize the HTTP client.

        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            debug: Enable debug logging
            max_retries: Maximum number of retry attempts (default: from env or 3)
            retry_delay: Base delay between retries in seconds (default: from env or 1.0)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"X-API-KEY": api_key}
        self.debug = debug

    @with_retry()
    def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Union[Dict, bytes]] = None,
        params: Optional[Dict] = None,
        stream: bool = False,
    ) -> Any:
        """
        Make a request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            stream: Whether to stream the response

        Returns:
            API response (dict or response object if streaming)

        Raises:
            requests.exceptions.RequestException: For network errors
            Exception: For API errors
        """
        url = f"{self.base_url}{endpoint}"

        headers = self.headers.copy()
        if data is not None and not isinstance(data, bytes):
            headers["Content-Type"] = "application/json"

        if self.debug:
            print(f"Request: {method} {url}")
            print(f"Headers: {headers}")
            if params:
                print(f"Params: {params}")
            if data and not isinstance(data, bytes):
                print(f"Data: {json.dumps(data, indent=2)}")

        if isinstance(data, bytes):
            # For binary data (like file uploads)
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                params=params,
                stream=stream,
            )
        else:
            # For JSON data
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                stream=stream,
            )

        if self.debug:
            print(f"Response status: {response.status_code}")
            if not stream:
                print(f"Response headers: {dict(response.headers)}")
                try:
                    print(f"Response body: {json.dumps(response.json(), indent=2)}")
                except:
                    print(f"Response body: {response.text}")

        # Check for errors
        if response.status_code >= 400:
            error_msg = f"API request failed with status {response.status_code}"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_msg = f"{error_msg}: {error_data['detail']}"
            except:
                if response.text:
                    error_msg = f"{error_msg}: {response.text}"

            # Raise specific exceptions based on status code
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key or unauthorized access")
            else:
                raise APIError(error_msg, status_code=response.status_code)

        # Return streaming response as-is
        if stream:
            return response

        # Return JSON response, or text if not JSON
        try:
            return response.json()
        except:
            return {"text": response.text}

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            API response
        """
        return self.request("GET", endpoint, params=params)

    def post(
        self,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint
            data: Request body data
            params: Query parameters

        Returns:
            API response
        """
        return self.request("POST", endpoint, data=data, params=params)

    @with_retry()
    def post_streaming(
        self, endpoint: str, data: bytes, params: Optional[Dict] = None
    ) -> Iterator[Dict]:
        """
        Make a streaming POST request for file uploads.

        Args:
            endpoint: API endpoint
            data: Binary data to upload
            params: Query parameters

        Yields:
            Parsed SSE messages

        Raises:
            Exception: For upload errors
        """
        response = self.request("POST", endpoint, data=data, params=params, stream=True)

        # Check initial response status
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", response.text)
            except:
                pass
            raise Exception(
                f"Upload failed with status {response.status_code}: {error_detail}"
            )

        # Process Server-Sent Events
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    try:
                        message_data = json.loads(line_str[6:])
                        yield message_data
                    except json.JSONDecodeError:
                        if self.debug:
                            print(f"Failed to parse SSE message: {line_str}")
                        continue

    @with_retry()
    def post_file(
        self,
        endpoint: str,
        files: Dict[str, tuple],
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Make a POST request with file upload.

        Args:
            endpoint: API endpoint
            files: Files to upload in format {"field": ("filename", file_content, "content-type")}
            params: Query parameters

        Returns:
            API response
        """
        url = f"{self.base_url}{endpoint}"

        if self.debug:
            print(f"Request: POST {url}")
            print(f"Headers: {self.headers}")
            if params:
                print(f"Params: {params}")

        response = requests.post(
            url,
            params=params,
            files=files,
            headers=self.headers,
        )

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", response.text)
            except:
                pass
            raise Exception(
                f"File upload failed with status {response.status_code}: {error_detail}"
            )

        return response.json()

    @with_retry()
    def download_binary(self, endpoint: str) -> bytes:
        """
        Download binary data from an endpoint.

        Args:
            endpoint: API endpoint

        Returns:
            Binary data

        Raises:
            Exception: For download errors
        """
        url = f"{self.base_url}{endpoint}"

        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", response.text)
            except:
                pass
            raise Exception(
                f"Download failed with status {response.status_code}: {error_detail}"
            )

        return response.content

    @with_retry()
    def download_from_url(self, url: str) -> bytes:
        """
        Download binary data from a presigned URL.

        Args:
            url: Full URL to download from

        Returns:
            Binary data

        Raises:
            Exception: For download errors
        """
        # Don't send authentication headers for presigned URLs
        response = requests.get(url)

        if response.status_code != 200:
            error_detail = response.text
            raise Exception(
                f"Download failed with status {response.status_code}: {error_detail}"
            )

        return response.content
