#include "monarch/torch-sys-cuda/src/bridge_rocm.h"

namespace monarch {

std::unique_ptr<at::cuda::CUDAEvent>
create_hip_event(bool enable_timing, bool blocking, bool interprocess) {
  unsigned int flags = (blocking ? hipEventBlockingSync : hipEventDefault) |
      (enable_timing ? hipEventDefault : hipEventDisableTiming) |
      (interprocess ? hipEventInterprocess : hipEventDefault);

  return std::make_unique<at::cuda::CUDAEvent>(flags);
}

void record_event(at::cuda::CUDAEvent& event, const c10::hip::HIPStream& stream) {
  at::hip::HIPStreamMasqueradingAsCUDA masq_stream(stream);
  event.record(masq_stream);
}

void block_event(at::cuda::CUDAEvent& event, const c10::hip::HIPStream& stream) {
  at::hip::HIPStreamMasqueradingAsCUDA masq_stream(stream);
  event.block(masq_stream);
}

std::shared_ptr<c10::hip::HIPStream> get_current_hip_stream(
    c10::DeviceIndex device) {
  return std::make_shared<c10::hip::HIPStream>(
      c10::hip::getCurrentHIPStream(device));
}

std::shared_ptr<c10::hip::HIPStream> create_hip_stream(
    c10::DeviceIndex device,
    int32_t priority) {
  return std::make_shared<c10::hip::HIPStream>(
      c10::hip::getStreamFromPool((const int)priority, device));
}

void set_current_hip_stream(const c10::hip::HIPStream& stream) {
  auto device = c10::hip::current_device();
  if (device != stream.device_index()) {
    c10::hip::set_device(stream.device_index());
  }
  at::hip::setCurrentHIPStream(stream);
}

size_t get_stream_handle(const c10::hip::HIPStream& stream) {
  return reinterpret_cast<size_t>(stream.stream());
}

} // namespace monarch
