import 'package:flutter/material.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../controller/chat_controller.dart';

class ModelSelectorWidget extends StatelessWidget {
  final ChatController controller;

  const ModelSelectorWidget({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
      final isLoading = controller.isLoading.value;
      final backends = controller.backendsSelected.value;
      final progress = controller.downloadProgress.value;
      final status = controller.loadingStatus.value;

      return Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const Text(
            'Selecione o modelo Gemma para baixar/carregar:',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          ToggleButtons(
            onPressed: isLoading ? null : (index) => controller.selectBackend(index),
            borderRadius: const BorderRadius.all(Radius.circular(20)),
            selectedBorderColor: Colors.teal[700],
            selectedColor: Colors.white,
            fillColor: Colors.teal[200],
            color: Colors.teal[700],
            constraints: const BoxConstraints(
              minHeight: 28.0,
              minWidth: 80.0,
            ),
            isSelected: backends,
            children: const [
              Text('CPU'),
              Text('GPU'),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: isLoading ? null : () => controller.downloadAndLoadModel(true),
                child: const Text('Carregar 2B'),
              ),
              const SizedBox(width: 16),
              ElevatedButton(
                onPressed: isLoading ? null : () => controller.downloadAndLoadModel(false),
                child: const Text('Carregar 4B'),
              ),
            ],
          ),
          if (isLoading) ...[
            const SizedBox(height: 24),
            Text(
              status,
              style: const TextStyle(fontWeight: FontWeight.w500),
            ),
            const SizedBox(height: 8),
            LinearProgressIndicator(value: progress > 0 ? progress : null),
            const SizedBox(height: 8),
            Text('${(progress * 100).toStringAsFixed(1)}%'),
          ],
          const Divider(height: 40),
        ],
      );
    },
    );
  }
}
