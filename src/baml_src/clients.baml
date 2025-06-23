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
      "X-Title" "Kuzu Graph RAG" // Optional
    }
  }
}

client<llm> OpenRouterGPT4o {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "openai/gpt-4o"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu Graph RAG" // Optional
    }
  }
}

client<llm> OpenRouterGPT41 {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "openai/gpt-4.1-2025-04-14"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu Graph RAG" // Optional
    }
  }
}

client<llm> OpenRouterPhi4 {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "microsoft/phi-4"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu Graph RAG" // Optional
    }
  }
}

client<llm> OpenRouterQwen3_30b_A3b {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "qwen/qwen3-30b-a3b"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu Graph RAG" // Optional
    }
  }
}

client<llm> OpenRouterGoogleGemini25Flash {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "google/gemini-2.5-flash"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu Graph RAG" // Optional
    }
  }
}

client<llm> OpenRouterGoogleGemini2Flash {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "google/gemini-2.0-flash-001"
    temperature 0.1
    headers {
      "HTTP-Referer" "https://kuzudb.com" // Optional
      "X-Title" "Kuzu Graph RAG" // Optional
    }
  }
}

client<llm> Gemini2Flash {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0.1
    }
  }
}

client<llm> Gemini25Flash {
  provider google-ai
  options {
    model "gemini-2.5-flash"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0.1
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