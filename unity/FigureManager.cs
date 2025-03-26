using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class FigureManager : MonoBehaviour
{
    [SerializeField] private GameObject figurePrefab;
    [SerializeField] private Transform figureContainer;

    private List<(Figure figure, Texture2D texture)> pendingFigures = new List<(Figure, Texture2D)>();
    private HashSet<string> processedFigureFiles = new HashSet<string>();
    private List<GameObject> instantiatedFigures = new List<GameObject>();

    public bool HasProcessedFigure(string filename)
    {
        return processedFigureFiles.Contains(filename);
    }

    public void AddPendingFigure(Figure figure, Texture2D texture)
    {
        if (!processedFigureFiles.Contains(figure.filename))
        {
            processedFigureFiles.Add(figure.filename);
            pendingFigures.Add((figure, texture));
            Debug.Log($"Added pending figure: {figure.filename}");
        }
    }

    public void DisplayCollectedFigures()
    {
        // Sort pendingFigures by score in descending order
        var sortedFigures = pendingFigures
            .OrderByDescending(x => x.figure.score)
            .ToList();

        foreach (var (figure, texture) in sortedFigures)
        {
            GameObject newFigureObj = Instantiate(figurePrefab, figureContainer);
            instantiatedFigures.Add(newFigureObj);

            var figureDisplay = newFigureObj.GetComponent<FigureDisplay>();
            if (figureDisplay != null)
            {
                figureDisplay.SetContent(texture, figure.title, figure.score);
            }
            else
            {
                Debug.LogError("FigureDisplay component missing from prefab");
                Destroy(newFigureObj);
            }
        }

        pendingFigures.Clear();
    }

    public void ClearAllFigures()
    {
        foreach (var figure in instantiatedFigures)
        {
            Destroy(figure);
        }
        instantiatedFigures.Clear();
        pendingFigures.Clear();
        processedFigureFiles.Clear();
    }
}