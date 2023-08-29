# LangSync

Elasticsearch Vector Store synchronization.

The sync API is used to upload a JSON Lines file to the vector store.
The JSON Lines file should contain JSON objects that represent multiple pages with the following properties:

* **content** contains the page content as plain text that is text splitted and indexed to the vector store
* **source** contains the url to the original page used in the chatbot as cited source
* **title** contains the page title used to create nice looking links

```json
{
  "content": "plain text",
  "source": "https://example.com/",
  "title": "Page Title"
}
```

The API documentation is available at http://127.0.0.1:8080/docs.

Example file upload:

```curl -X 'POST' \
  'http://127.0.0.1:8080/sync?index_name=myindex' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'jsonl_file=@example.jsonl'
```
