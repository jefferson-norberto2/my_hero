import 'package:flutter/material.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_metrics.dart';
import 'package:my_hero/src/shared/models/compute_mode.dart';

/// Compact AppBar status pill displaying allocated memory, compute hardware tag,
/// and context window utilization.
class ChatMetricsBadge extends StatelessWidget {
  final ChatMetrics metrics;
  final VoidCallback onTap;

  const ChatMetricsBadge({
    super.key,
    required this.metrics,
    required this.onTap,
  });

  Color _contextStatusColor(double usagePercent) {
    if (usagePercent > 80) return Colors.red.shade400;
    if (usagePercent > 50) return Colors.amber.shade700;
    return Colors.teal;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final usagePercent = metrics.contextUsagePercent;
    final remainingPercent = metrics.remainingPercent;
    final badgeColor = _contextStatusColor(usagePercent);
    final computeMode = metrics.computeMode;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: theme.colorScheme.surfaceContainerHighest
                .withValues(alpha: 0.8),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: badgeColor.withValues(alpha: 0.5)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Tooltip(
                message: metrics.hardwareTooltip,
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.memory_rounded,
                      size: 14,
                      color: Colors.teal.shade700,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      metrics.memoryFormatted,
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    if (computeMode.isActive) ...[
                      const SizedBox(width: 5),
                      _ComputeModeTag(mode: computeMode),
                    ],
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Container(
                width: 1,
                height: 12,
                color: Colors.grey.withValues(alpha: 0.4),
              ),
              const SizedBox(width: 8),
              Tooltip(
                message: metrics.contextTooltip,
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.pie_chart_outline_rounded,
                      size: 14,
                      color: badgeColor,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'Ctx: ${usagePercent.toStringAsFixed(0)}% (restam ${remainingPercent.toStringAsFixed(0)}%)',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        color: badgeColor,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ComputeModeTag extends StatelessWidget {
  final ComputeMode mode;

  const _ComputeModeTag({required this.mode});

  @override
  Widget build(BuildContext context) {
    final _TagPalette palette = _paletteForMode(mode);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1.5),
      decoration: BoxDecoration(
        color: palette.background,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: palette.border),
      ),
      child: Text(
        mode.tagLabel,
        style: TextStyle(
          fontSize: 9.5,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.2,
          color: palette.foreground,
          height: 1.1,
        ),
      ),
    );
  }

  _TagPalette _paletteForMode(ComputeMode mode) {
    switch (mode) {
      case ComputeMode.gpu:
        return const _TagPalette(
          background: Color(0xFFD8F3DC),
          foreground: Color(0xFF1B4332),
          border: Color(0xFF95D5B2),
        );
      case ComputeMode.cpu:
        return const _TagPalette(
          background: Color(0xFFE2E8F0),
          foreground: Color(0xFF334155),
          border: Color(0xFFCBD5E1),
        );
      case ComputeMode.hybrid:
        return const _TagPalette(
          background: Color(0xFFFEF3C7),
          foreground: Color(0xFF78350F),
          border: Color(0xFFFCD34D),
        );
      case ComputeMode.none:
        return const _TagPalette(
          background: Color(0xFFE5E7EB),
          foreground: Color(0xFF374151),
          border: Color(0xFFD1D5DB),
        );
    }
  }
}

class _TagPalette {
  final Color background;
  final Color foreground;
  final Color border;

  const _TagPalette({
    required this.background,
    required this.foreground,
    required this.border,
  });
}
