class ChatMetrics {
  final int promptTokens;
  final int evalTokens;
  final int totalTokens;
  final int contextLimit;
  final int vramBytes;

  const ChatMetrics({
    this.promptTokens = 0,
    this.evalTokens = 0,
    this.contextLimit = 4096,
    this.vramBytes = 0,
  }) : totalTokens = promptTokens + evalTokens;

  double get contextUsagePercent {
    if (contextLimit <= 0) return 0.0;
    return ((totalTokens / contextLimit) * 100).clamp(0.0, 100.0);
  }

  double get remainingPercent {
    return (100.0 - contextUsagePercent).clamp(0.0, 100.0);
  }

  int get remainingTokens {
    final remaining = contextLimit - totalTokens;
    return remaining > 0 ? remaining : 0;
  }

  String get vramFormatted {
    if (vramBytes <= 0) return '0 MB';
    final gb = vramBytes / (1024 * 1024 * 1024);
    if (gb >= 1.0) {
      return '${gb.toStringAsFixed(1)} GB';
    }
    final mb = vramBytes / (1024 * 1024);
    return '${mb.toStringAsFixed(0)} MB';
  }

  ChatMetrics copyWith({
    int? promptTokens,
    int? evalTokens,
    int? contextLimit,
    int? vramBytes,
  }) {
    return ChatMetrics(
      promptTokens: promptTokens ?? this.promptTokens,
      evalTokens: evalTokens ?? this.evalTokens,
      contextLimit: contextLimit ?? this.contextLimit,
      vramBytes: vramBytes ?? this.vramBytes,
    );
  }
}
