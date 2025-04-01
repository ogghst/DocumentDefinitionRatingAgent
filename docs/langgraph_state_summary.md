# LangGraph State and LLM Context Summary

This document summarizes the information managed within the LangGraph `GraphState` and the context available to the Large Language Model (LLM) at different stages of the analysis process, based on `graph_nodes.py` and `rag_engine.py`.

## LangGraph State (`GraphState`)

The `GraphState` is a dictionary-like object (TypedDict) that carries information throughout the graph execution. Here are the key fields observed and the nodes that primarily populate or modify them:

*   **`checklist_path`**: `str`
    *   **Populated by**: Initial input to the graph.
    *   **Used by**: `load_and_filter_checklist`
*   **`document_path`**: `str`
    *   **Populated by**: Initial input to the graph.
    *   **Used by**: `load_index_document`
*   **`target_phase`**: `str`
    *   **Populated by**: Initial input to the graph.
    *   **Used by**: `load_and_filter_checklist`
*   **`conversation_id`**: `Optional[str]`
    *   **Populated by**: Initial input to the graph (likely).
    *   **Used by**: Various nodes for logging/messaging (`send_message`), `analyze_checks_map_function` (passed via config metadata), `format_final_output`.
*   **`callback_manager`**: `Optional[Any]` (e.g., `WebsocketCallbackManager`)
    *   **Populated by**: Initial input to the graph (likely).
    *   **Used by**: `analyze_checks_map_function` (passed via config metadata to `analyze_check_rag`), `analyze_check_rag` (for streaming LLM output and getting user input).
*   **`checks_for_phase`**: `List[CheckItem]`
    *   **Populated by**: `load_and_filter_checklist`
    *   **Used by**: `decide_after_indexing` (conditional edge logic), `analyze_checks_map_function` (iterated over).
*   **`retriever`**: `Optional[VectorStoreRetriever]`
    *   **Populated by**: `load_index_document`
    *   **Used by**: `analyze_checks_map_function` (passed via config metadata to `analyze_check_rag`), `analyze_check_rag` (invoked by the RAG chain).
*   **`analysis_results`**: `List[CheckResult]`
    *   **Populated by**: `analyze_checks_map_function` (collects results from `analyze_check_rag`).
    *   **Used by**: `format_final_output`.
*   **`final_results`**: `Optional[List[Dict[str, Any]]]`
    *   **Populated by**: `format_final_output` (formats `analysis_results` for output).
    *   **Used by**: Potentially downstream processes after the graph finishes.
*   **`error_message`**: `Optional[str]`
    *   **Populated by**: Any node encountering an error.
    *   **Used by**: `decide_after_indexing` (conditional edge logic), potentially checked at the start of nodes to skip processing.

## LLM Context

The LLM interacts with the system in two main ways: analyzing checklist items and generating follow-up questions.

1.  **Checklist Item Analysis (`analyze_check_rag` via `create_hybrid_rag_chain`)**:
    *   **Triggered by**: `analyze_checks_map_function` invoking `analyze_check_rag` for each `CheckItem` in `checks_for_phase`.
    *   **Input passed to RAG Chain**:
        *   `check_id`, `check_name`, `check_description`: From the specific `CheckItem` being analyzed.
        *   `retriever`: From the `GraphState` (passed via `config`). Used to fetch relevant document chunks based on `check_description`.
        *   `additional_info`: Text provided by the user during interactive review loops within `analyze_check_rag`. Initially empty.
    *   **LLM Prompt Context (`analysis_prompt_template`)**: The LLM receives a prompt containing:
        *   The detailed `CheckItem` information (`id`, `name`, `description`).
        *   The content of the document chunks retrieved by the `retriever` (formatted as `context`).
        *   Any `additional_info` provided by the user.
        *   Specific instructions on how to analyze the context against the checklist item and the required JSON output format (`format_instructions`).

2.  **Follow-up Question Generation (`generate_questions`)**:
    *   **Triggered by**: `analyze_check_rag` when the analysis reliability score is low (e.g., < 50) and user input is needed.
    *   **Input passed to LLM**:
        *   `check_id`, `check_name`, `check_description`: From the specific `CheckItem`.
        *   `current_reliability`, `is_met_status`, `analysis_details`: From the `CheckResult` of the current low-confidence analysis.
        *   `sources_summary`: A summary of the evidence (`sources`) found so far.
    *   **LLM Prompt Context (`question_prompt_template`)**: The LLM receives a prompt containing:
        *   The `CheckItem` details.
        *   The current (low-reliability) analysis status and reasoning.
        *   A summary of the evidence found so far.
        *   Instructions to generate 1-3 specific questions to address information gaps.

## State Access in RAG Engine

The `rag_engine.py` functions (`create_hybrid_rag_chain`, `generate_questions`) do not directly access the `GraphState`. Instead, the necessary information (like the `retriever`, `check_item` details, `conversation_id`, `callback_manager`) is explicitly passed into the relevant functions (`analyze_check_rag`) or chain invocations (`hybrid_rag_chain.ainvoke`) using a `RunnableConfig` object, often populated within the `analyze_checks_map_function` node. This keeps the RAG engine decoupled from the specific graph state structure. 