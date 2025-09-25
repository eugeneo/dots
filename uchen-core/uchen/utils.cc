
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "absl/log/log.h"

namespace uchen::utils {

void PrintFrames(size_t depth = 3, size_t skip = 1) {
  void* pcs[20];
  int sizes[20];
  int frames = absl::GetStackFrames(pcs, sizes, depth, skip);
  char frame[20000];
  LOG(INFO) << "---------------------------------";
  for (int i = 0; i < frames; ++i) {
    if (absl::Symbolize(pcs[i], frame, sizeof(frame))) {
      LOG(INFO) << frame << " size: " << sizes[i];
    }
  }
}

}  // namespace uchen::utils