// nvmoe-trace — capture MoE expert-routing decisions from a llama.cpp run.
//
// Registers a scheduler eval callback that observes the "ffn_moe_topk-<il>"
// tensors (the selected expert ids per token, per MoE layer) and appends one
// raw JSONL record per observation:
//     {"l": <layer>, "t": <n_tokens>, "e": [[ids...], [ids...], ...]}
// Post-process into the nvmoe simulator's per-token format with
// sim/trace_post.py from the nvmoe repo.
//
// Usage:
//   llama-nvmoe-trace -m model.gguf -o trace.raw.jsonl [-n n_predict]
//                     [-ngl n_gpu_layers] [-f prompt.txt | prompt words]
//
// Based on examples/simple/simple.cpp (MIT).

#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct trace_state {
    FILE * out = nullptr;
    long   records = 0;
};

// eval callback: ask-phase selects which tensors to observe, data-phase logs them
static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool is_topk = strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (ask) {
        return is_topk; // request data only for expert-selection tensors
    }
    if (!is_topk) {
        return true;
    }

    auto * state = (trace_state *) user_data;
    if (t->type != GGML_TYPE_I32) {
        fprintf(stderr, "nvmoe-trace: unexpected type %d for %s\n", (int) t->type, t->name);
        return true;
    }

    const int top_k    = (int) t->ne[0];
    const int n_tokens = (int) t->ne[1];

    // layer index is the suffix after "ffn_moe_topk-"
    const char * dash = strrchr(t->name, '-');
    const int layer = dash ? atoi(dash + 1) : -1;

    std::vector<int32_t> ids((size_t) top_k * n_tokens);
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));

    fprintf(state->out, "{\"l\":%d,\"t\":%d,\"e\":[", layer, n_tokens);
    for (int tok = 0; tok < n_tokens; tok++) {
        fprintf(state->out, "%s[", tok ? "," : "");
        for (int k = 0; k < top_k; k++) {
            fprintf(state->out, "%s%d", k ? "," : "", ids[(size_t) tok * top_k + k]);
        }
        fprintf(state->out, "]");
    }
    fprintf(state->out, "]}\n");
    state->records++;
    return true;
}

static void print_usage(int, char ** argv) {
    printf("\nusage:\n");
    printf("\n    %s -m model.gguf -o trace.jsonl [-n n_predict] [-ngl layers] [-f prompt.txt | prompt]\n\n", argv[0]);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string trace_path = "trace.raw.jsonl";
    std::string prompt     = "Hello my name is";
    int ngl       = 0;   // CPU by default: tracing only needs the model to run
    int n_predict = 256;

    for (int i = 1; i < argc; i++) {
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { print_usage(argc, argv); exit(1); }
            (void) flag;
            return argv[++i];
        };
        if      (strcmp(argv[i], "-m")   == 0) { model_path = need("-m"); }
        else if (strcmp(argv[i], "-o")   == 0) { trace_path = need("-o"); }
        else if (strcmp(argv[i], "-n")   == 0) { n_predict  = atoi(need("-n")); }
        else if (strcmp(argv[i], "-ngl") == 0) { ngl        = atoi(need("-ngl")); }
        else if (strcmp(argv[i], "-f")   == 0) {
            std::ifstream fh(need("-f"));
            if (!fh) { fprintf(stderr, "cannot open prompt file\n"); return 1; }
            std::stringstream ss; ss << fh.rdbuf(); prompt = ss.str();
        }
        else {
            prompt = argv[i];
            for (i++; i < argc; i++) { prompt += " "; prompt += argv[i]; }
        }
    }
    if (model_path.empty()) { print_usage(argc, argv); return 1; }

    trace_state state;
    state.out = fopen(trace_path.c_str(), "w");
    if (!state.out) { fprintf(stderr, "cannot open %s\n", trace_path.c_str()); return 1; }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) { fprintf(stderr, "unable to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize the prompt\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_prompt + n_predict;
    ctx_params.n_batch = n_prompt;
    // the whole point: observe routing tensors during graph execution
    ctx_params.cb_eval           = trace_cb;
    ctx_params.cb_eval_user_data = &state;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) { fprintf(stderr, "failed to create the llama_context\n"); return 1; }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    const auto t_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) { fprintf(stderr, "failed to eval\n"); return 1; }
        n_pos += batch.n_tokens;

        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) { fwrite(buf, 1, n, stdout); fflush(stdout); }

        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode++;
    }

    const auto t_end = ggml_time_us();
    fprintf(stderr, "\nnvmoe-trace: %d prompt + %d decoded tokens, %.2f tok/s, %ld routing records -> %s\n",
            n_prompt, n_decode, n_decode / ((t_end - t_start) / 1000000.0), state.records, trace_path.c_str());

    fclose(state.out);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
