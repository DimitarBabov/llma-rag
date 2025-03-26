using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using System;
using TMPro;
using System.Text;

public class RAGClient : MonoBehaviour
{
    [SerializeField] private string apiUrl = "http://localhost:5001/api";
    [SerializeField] private TMP_InputField queryInput;
    [SerializeField] private TextMeshProUGUI responseText;
    [SerializeField] private FigureManager figureManager;

    private bool isStreaming = false;
    private StringBuilder currentResponse = new StringBuilder();

    [System.Serializable]
    private class QueryRequest
    {
        public string query;
    }

    [System.Serializable]
    private class StreamChunk
    {
        public string text;
        public List<Figure> figures;
    }

    public void SendQuery()
    {
        if (string.IsNullOrEmpty(queryInput.text) || isStreaming) return;
        StartCoroutine(StreamQueryCoroutine(queryInput.text));
    }

    private IEnumerator StreamQueryCoroutine(string query)
    {
        isStreaming = true;
        currentResponse.Clear();
        responseText.text = "";
        figureManager.ClearAllFigures();

        yield return StartCoroutine(SendRequest(query));

        isStreaming = false;
    }

    private IEnumerator SendRequest(string query)
    {
        var request = new QueryRequest { query = query };
        string jsonData = JsonUtility.ToJson(request);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);

        using (UnityWebRequest www = new UnityWebRequest($"{apiUrl}/generator", "POST"))
        {
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.SetRequestHeader("Accept", "text/event-stream");

            var operation = www.SendWebRequest();
            int lastProcessedLength = 0;

            while (!operation.isDone)
            {
                if (www.downloadHandler.data != null)
                {
                    string newData = System.Text.Encoding.UTF8.GetString(www.downloadHandler.data);
                    if (newData.Length > lastProcessedLength)
                    {
                        ProcessStreamData(newData.Substring(lastProcessedLength));
                        lastProcessedLength = newData.Length;
                    }
                }
                yield return new WaitForSeconds(0.1f);
            }

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"Error: {www.error}");
                responseText.text = "Error occurred while fetching response.";
            }
            else
            {
                figureManager.DisplayCollectedFigures();
            }
        }
    }

    private void ProcessStreamData(string data)
    {
        string[] lines = data.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string line in lines)
        {
            if (line.StartsWith("data: "))
            {
                string jsonData = line.Substring(6);
                try
                {
                    if (jsonData.Trim() == "[DONE]")
                    {
                        Debug.Log("Stream completed");
                        continue;
                    }

                    StreamChunk chunk = JsonUtility.FromJson<StreamChunk>(jsonData);

                    if (!string.IsNullOrEmpty(chunk.text))
                    {
                        currentResponse.Append(chunk.text);
                        responseText.text = currentResponse.ToString();
                    }

                    if (chunk.figures != null && chunk.figures.Count > 0)
                    {
                        foreach (var figure in chunk.figures)
                        {
                            if (!figureManager.HasProcessedFigure(figure.filename))
                            {
                                StartCoroutine(LoadFigure(figure));
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"Error parsing stream data: {e.Message}\nData: {jsonData}");
                }
            }
        }
    }

    private IEnumerator LoadFigure(Figure figure)
    {
        // Remove 'api' from the path for static files
        string baseUrl = apiUrl.Replace("/api", "");
        string figureUrl = $"{baseUrl}/static/{figure.filename}";

        using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(figureUrl))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Texture2D texture = DownloadHandlerTexture.GetContent(www);
                if (texture != null)
                {
                    figureManager.AddPendingFigure(figure, texture);
                }
                else
                {
                    Debug.LogError($"Failed to create texture for figure: {figure.title}");
                }
            }
            else
            {
                Debug.LogError($"Error loading figure: {www.error}. URL: {figureUrl}");
            }
        }
    }

    // Optional: Check server health on start
    private IEnumerator Start()
    {
        using (UnityWebRequest www = UnityWebRequest.Get($"{apiUrl}/health"))
        {
            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Failed to connect to RAG server. Please ensure it's running.");
            }
        }
    }
}
