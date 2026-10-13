# Installing ep-mcp (for AI agents such as Cline)

ep-mcp serves one or more ExpertPacks (folders of Markdown knowledge) over MCP via stdio.

## Prerequisites
- [uv](https://docs.astral.sh/uv/) installed (`uvx` on PATH), Python 3.11+.
- An ExpertPack folder on disk (example: clone https://github.com/brianhearn/expert-pack and use one of its packs).
- An embedding API key: `GEMINI_API_KEY` (default) or `AZURE_OPENAI_API_KEY`.

## Steps
1. Create a config YAML, e.g. `~/ep-mcp/config.yaml`, based on
   https://github.com/brianhearn/ep-mcp/blob/main/.examples/single-pack-stdio-config.yaml.
   Set the pack `path` to the absolute path of the ExpertPack folder.
2. Ask the user for their Gemini API key (or Azure OpenAI key). Never write it into the repo.
3. Add this to the MCP settings file:

```json
{
  "mcpServers": {
    "ep-mcp": {
      "command": "uvx",
      "args": ["ep-mcp", "serve", "--config", "/ABSOLUTE/PATH/config.yaml", "--transport", "stdio"],
      "env": { "GEMINI_API_KEY": "<user's key>" }
    }
  }
}
```
4. Restart the MCP client. The first start indexes the pack and may take a minute.
5. Verify: the tools `ep_search`, `ep_read`, `ep_list_topics`, and `ep_graph_traverse` appear. Try `ep_list_topics`.
