// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> OpenRouterGPT4oMini {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "openai/gpt-4o-mini"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu" // Optional
    }
  }
}

client<llm> OpenRouterGPT4o {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "openai/gpt-4o"
    temperature 0.0
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu" // Optional
    }
  }
}

client<llm> OpenRouterClaude37Sonnet {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "anthropic/claude-3.7-sonnet"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu" // Optional
    }
  }
}

client<llm> OpenRouterClaude35Sonnet {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "anthropic/claude-3.5-sonnet"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu" // Optional
    }
  }
}

client<llm> OpenRouterGPT4oMiniGenerate {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "openai/gpt-4o-mini"
    temperature 0.3
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu" // Optional
    }
  }
}

client<llm> Gemini2Flash {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0.0
    }
  }
}

client<llm> GTP4oMiniExtract {
  provider openai
  options {
    model "gpt-4o-mini"
    api_key env.OPENAI_API_KEY
    temperature 0.0
  }
}

client<llm> GTP4oMiniGenerate {
  provider openai
  options {
    model "gpt-4o-mini"
    api_key env.OPENAI_API_KEY
    temperature 0.3
  }
}

client<llm> GPT4o {
  provider openai
  options {
    model "gpt-4o"
    api_key env.OPENAI_API_KEY
    temperature 0.0
  }
}

client<llm> FastOpenAI {
  provider openai
  options {
    model "gpt-4o-mini"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> Fast {
  provider round-robin
  options {
    // This will alternate between the two clients
    // Learn more at https://docs.boundaryml.com/docs/snippets/clients/round-robin
    strategy [OpenRouterGPT4oMini, FastOpenAI]
  }
}

client<llm> OpenaiFallback {
  provider fallback
  options {
    // This will try the clients in order until one succeeds
    // Learn more at https://docs.boundaryml.com/docs/snippets/clients/fallback
    strategy [GTP4oMiniGenerate, GPT4o]
  }
}

// Custom LLM inference server by YourTechBud
client<llm> Inferix_Gemma3_27b {
  provider openai-generic
  options {
    base_url "https://inferix.yourtechbud.studio/inferix/v1/llm"
    api_key env.INFERIX_API_KEY
    model "gemma3:27b"
    temperature 0.1
  }
}

client<llm> OllamaGemma3_12b {
  provider openai-generic
  options {
    base_url "http://localhost:11434/v1"
    model "gemma3:12b"
    temperature 0.1
  }
}
