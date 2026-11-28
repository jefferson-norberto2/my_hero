package com.example.my_hero

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.StatFs
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private var inferenceChannel: MethodChannel? = null
    private var resourcesChannel: MethodChannel? = null
    private var hasRequestedNotificationPermission: Boolean = false

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        val channel = MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            INFERENCE_CHANNEL_NAME,
        )
        inferenceChannel = channel

        val devResourcesChannel = MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            DEVICE_RESOURCES_CHANNEL_NAME,
        )
        resourcesChannel = devResourcesChannel

        devResourcesChannel.setMethodCallHandler { call, result ->
            when (call.method) {
                "getDeviceResources" -> {
                    val activityManager =
                        getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
                    val memInfo = ActivityManager.MemoryInfo()
                    activityManager?.getMemoryInfo(memInfo)

                    val freeStorageBytes = try {
                        StatFs(applicationContext.filesDir.path).availableBytes
                    } catch (_: Exception) {
                        0L
                    }

                    result.success(
                        hashMapOf<String, Any>(
                            "totalRamBytes" to memInfo.totalMem,
                            "availableRamBytes" to memInfo.availMem,
                            "lowMemory" to memInfo.lowMemory,
                            "memoryThresholdBytes" to memInfo.threshold,
                            "freeStorageBytes" to freeStorageBytes,
                        ),
                    )
                }

                else -> result.notImplemented()
            }
        }

        InferenceForegroundService.onStopRequestedListener = {
            runOnUiThread {
                inferenceChannel?.invokeMethod("onStopRequested", null)
            }
        }

        channel.setMethodCallHandler { call, result ->
            when (call.method) {
                "start" -> {
                    val title = call.argument<String>("title") ?: "My Hero"
                    val body = call.argument<String>("body") ?: "Gerando resposta..."
                    requestNotificationPermissionIfNeeded()
                    InferenceForegroundService.start(applicationContext, title, body)
                    result.success(null)
                }

                "update" -> {
                    val title = call.argument<String>("title") ?: "My Hero"
                    val body = call.argument<String>("body") ?: "Gerando resposta..."
                    InferenceForegroundService.updateNotification(applicationContext, title, body)
                    result.success(null)
                }

                "stop" -> {
                    val notifyIfBackgrounded =
                        call.argument<Boolean>("notifyIfBackgrounded") ?: false
                    val completedTitle = call.argument<String>("completedTitle")
                    val completedBody = call.argument<String>("completedBody")
                    InferenceForegroundService.stopAndMaybeNotify(
                        context = applicationContext,
                        notifyIfBackgrounded = notifyIfBackgrounded,
                        completedTitle = completedTitle,
                        completedBody = completedBody,
                    )
                    result.success(null)
                }

                "clearCompletedNotification" -> {
                    InferenceForegroundService.clearCompletedNotification(applicationContext)
                    result.success(null)
                }

                else -> result.notImplemented()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        InferenceForegroundService.clearCompletedNotification(applicationContext)
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        InferenceForegroundService.onStopRequestedListener = null
        inferenceChannel?.setMethodCallHandler(null)
        inferenceChannel = null
        resourcesChannel?.setMethodCallHandler(null)
        resourcesChannel = null
        super.cleanUpFlutterEngine(flutterEngine)
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (hasRequestedNotificationPermission) return
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        hasRequestedNotificationPermission = true
        requestPermissions(
            arrayOf(Manifest.permission.POST_NOTIFICATIONS),
            REQUEST_CODE_POST_NOTIFICATIONS,
        )
    }

    companion object {
        private const val INFERENCE_CHANNEL_NAME =
            "com.example.my_hero/inference_notification"
        private const val DEVICE_RESOURCES_CHANNEL_NAME =
            "com.example.my_hero/device_resources"
        private const val REQUEST_CODE_POST_NOTIFICATIONS = 4201
    }
}
