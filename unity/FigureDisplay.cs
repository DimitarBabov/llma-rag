using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class FigureDisplay : MonoBehaviour
{
    [SerializeField] private RawImage figureImage;
    [SerializeField] private TextMeshProUGUI titleText;
    [SerializeField] private TextMeshProUGUI relevanceScoreText;
    [SerializeField] public float relevanceScore; 

    public void SetContent(Texture2D texture, string title, float score)
    {
        if (texture != null)
        {
            figureImage.texture = texture;
            titleText.text = $"{title}";
            relevanceScoreText.text = $"(Relevance Score: {score:F2})";
            relevanceScore = score; 
        }
    }

    private void OnDestroy()
    {
        // Clean up the texture when the object is destroyed
        if (figureImage.texture != null)
        {
            Destroy(figureImage.texture);
        }
    }
}