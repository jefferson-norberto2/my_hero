import 'package:flutter/material.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
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

  Color _contextStatusColor(AppThemeColors appColors, double usagePercent) {
    if (usagePercent > 80) return appColors.contextHigh;
    if (usagePercent > 50) return appColors.contextMedium;
    return appColors.contextLow;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final appColors = AppThemeColors.of(context);
    final usagePercent = metrics.contextUsagePercent;
    final badgeColor = _contextStatusColor(appColors, usagePercent);
    final computeMode = metrics.computeMode;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: appColors.badgeSurface,
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
                      color: appColors.infoIcon,
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
                color: theme.colorScheme.outlineVariant.withValues(alpha: 0.45),
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
                      'Ctx: ${usagePercent.toStringAsFixed(0)}%',
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
    final appColors = AppThemeColors.of(context);
    final _TagPalette palette = _paletteForMode(appColors, mode);

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

  _TagPalette _paletteForMode(AppThemeColors appColors, ComputeMode mode) {
    switch (mode) {
      case ComputeMode.gpu:
        return _TagPalette(
          background: appColors.gpuTagBackground,
          foreground: appColors.gpuTagForeground,
          border: appColors.gpuTagBorder,
        );
      case ComputeMode.cpu:
        return _TagPalette(
          background: appColors.cpuTagBackground,
          foreground: appColors.cpuTagForeground,
          border: appColors.cpuTagBorder,
        );
      case ComputeMode.hybrid:
        return _TagPalette(
          background: appColors.hybridTagBackground,
          foreground: appColors.hybridTagForeground,
          border: appColors.hybridTagBorder,
        );
      case ComputeMode.none:
        return _TagPalette(
          background: appColors.neutralTagBackground,
          foreground: appColors.neutralTagForeground,
          border: appColors.neutralTagBorder,
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
