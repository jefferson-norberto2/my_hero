/// Lifecycle status of an Ollama model pull (`/api/pull`) operation.
enum ModelDownloadStatus {
  idle,
  downloading,
  reconnecting,
  completed,
  cancelled,
  error;

  /// Whether a model download or reconnection attempt is currently active.
  bool get isActive =>
      this == ModelDownloadStatus.downloading ||
      this == ModelDownloadStatus.reconnecting;
}

/// Immutable state snapshot representing an Ollama model download progress.
class ModelDownloadState {
  final String modelName;
  final ModelDownloadStatus status;
  final String statusMessage;
  final int completedBytes;
  final int totalBytes;
  final int retryAttempt;
  final int maxRetries;
  final String? errorMessage;

  const ModelDownloadState({
    required this.modelName,
    required this.status,
    this.statusMessage = '',
    this.completedBytes = 0,
    this.totalBytes = 0,
    this.retryAttempt = 0,
    this.maxRetries = 3,
    this.errorMessage,
  });

  /// Initial idle state with no active download.
  const ModelDownloadState.idle()
      : modelName = '',
        status = ModelDownloadStatus.idle,
        statusMessage = '',
        completedBytes = 0,
        totalBytes = 0,
        retryAttempt = 0,
        maxRetries = 3,
        errorMessage = null;

  /// Whether this download is actively running or attempting reconnection.
  bool get isActive => status.isActive;

  /// Normalized progress in `[0.0, 1.0]`, or `0.0` if [totalBytes] is unknown.
  double get progress {
    if (totalBytes <= 0) return 0.0;
    return (completedBytes / totalBytes).clamp(0.0, 1.0);
  }

  /// Rounded percentage in `[0, 100]`, or `null` when [totalBytes] is unknown.
  int? get progressPercent {
    if (totalBytes <= 0) return null;
    return (progress * 100).round().clamp(0, 100);
  }

  ModelDownloadState copyWith({
    String? modelName,
    ModelDownloadStatus? status,
    String? statusMessage,
    int? completedBytes,
    int? totalBytes,
    int? retryAttempt,
    int? maxRetries,
    String? errorMessage,
    bool clearError = false,
  }) {
    return ModelDownloadState(
      modelName: modelName ?? this.modelName,
      status: status ?? this.status,
      statusMessage: statusMessage ?? this.statusMessage,
      completedBytes: completedBytes ?? this.completedBytes,
      totalBytes: totalBytes ?? this.totalBytes,
      retryAttempt: retryAttempt ?? this.retryAttempt,
      maxRetries: maxRetries ?? this.maxRetries,
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is ModelDownloadState &&
        other.modelName == modelName &&
        other.status == status &&
        other.statusMessage == statusMessage &&
        other.completedBytes == completedBytes &&
        other.totalBytes == totalBytes &&
        other.retryAttempt == retryAttempt &&
        other.maxRetries == maxRetries &&
        other.errorMessage == errorMessage;
  }

  @override
  int get hashCode => Object.hash(
        modelName,
        status,
        statusMessage,
        completedBytes,
        totalBytes,
        retryAttempt,
        maxRetries,
        errorMessage,
      );
}
