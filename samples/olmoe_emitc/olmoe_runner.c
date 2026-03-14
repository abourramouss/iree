// OLMoE native inference via EmitC module.
// Usage: olmoe_runner <params.irpa> <max_tokens> [token_ids...]
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "iree/runtime/api.h"
#include "olmoe_emitc.h"

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <params.irpa> <max_tokens> [token_ids...]\n", argv[0]);
    return 1;
  }
  const char* params_path = argv[1];
  int max_tokens = atoi(argv[2]);

  // System allocator
  extern iree_status_t iree_allocator_libc_ctl(
      void*, iree_allocator_command_t, const void*, void**);
  iree_allocator_t ha;
  ha.self = NULL;
  ha.ctl = iree_allocator_libc_ctl;

  // Create instance with all drivers
  iree_runtime_instance_options_t opts;
  iree_runtime_instance_options_initialize(&opts);
  iree_runtime_instance_options_use_all_available_drivers(&opts);
  iree_runtime_instance_t* instance = NULL;
  iree_status_t status = iree_runtime_instance_create(&opts, ha, &instance);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "Failed to create instance\n");
    iree_status_fprint(stderr, status);
    return 1;
  }

  // Create native module
  iree_vm_module_t* emitc_module = NULL;
  status = llm_inference_create(
      iree_runtime_instance_vm_instance(instance), ha, &emitc_module);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "Failed to create EmitC module\n");
    iree_status_fprint(stderr, status);
    return 1;
  }

  fprintf(stderr, "[native] EmitC module: %.*s\n",
      (int)iree_vm_module_name(emitc_module).size,
      iree_vm_module_name(emitc_module).data);

  // Just verify it loads — full inference needs more session setup
  fprintf(stderr, "[native] Module loaded successfully!\n");
  fprintf(stderr, "[native] Full inference runner needs proper session/device setup.\n");
  fprintf(stderr, "[native] Use the Python path for now:\n");
  fprintf(stderr, "  PYTHONPATH=~/iree-python-runtime:~/iree-python-compiler \\\n");
  fprintf(stderr, "  python3.11 ~/frank-models-olmoe-opt/scripts/chat_olmoe.py \\\n");
  fprintf(stderr, "    --params ~/models/olmoe-1b-7b-f16-fused.irpa \\\n");
  fprintf(stderr, "    --vmfb ~/models/olmoe_best.vmfb \\\n");
  fprintf(stderr, "    --backend cuda --model-dir ~/models/olmoe-1b-7b\n");

  iree_vm_module_release(emitc_module);
  iree_runtime_instance_release(instance);
  return 0;
}
