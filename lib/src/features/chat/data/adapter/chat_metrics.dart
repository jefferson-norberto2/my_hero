import 'package:my_hero/src/shared/models/compute_mode.dart';
import 'package:my_hero/src/shared/models/memory_allocation.dart';

/// Tracks token usage, context limit, and hardware memory consumption for active chats.
class ChatMetrics {
  final int promptTokens;
  final int evalTokens;
  final int totalTokens;
  final int contextLimit;
  final MemoryAllocation memoryAllocation;

  const ChatMetrics({
    this.promptTokens = 0,
    this.evalTokens = 0,
    this.contextLimit = 4096,
    this.memoryAllocation = const MemoryAllocation.empty(),
  }) : totalTokens = promptTokens + evalTokens;

  ComputeMode get computeMode => memoryAllocation.computeMode;

  int get vramBytes => memoryAllocation.vramBytes;

  int get ramBytes => memoryAllocation.ramBytes;

  int get totalMemoryBytes => memoryAllocation.totalBytes;

  int get displayedMemoryBytes => memoryAllocation.displayedBytes;

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

  String get memoryFormatted => memoryAllocation.formattedMemory;

  String get vramFormatted => memoryAllocation.formattedVram;

  String get ramFormatted => memoryAllocation.formattedRam;

  String get hardwareTooltip => memoryAllocation.hardwareTooltip;

  ChatMetrics copyWith({
    int? promptTokens,
    int? evalTokens,
    int? contextLimit,
    MemoryAllocation? memoryAllocation,
  }) {
    return ChatMetrics(
      promptTokens: promptTokens ?? this.promptTokens,
      evalTokens: evalTokens ?? this.evalTokens,
      contextLimit: contextLimit ?? this.contextLimit,
      memoryAllocation: memoryAllocation ?? this.memoryAllocation,
    );
  }
}
