using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class FigureDisplay : MonoBehaviour
{
    [SerializeField] private RawImage figureImage;
    [SerializeField] private TextMeshProUGUI titleText;
    [SerializeField] private TextMeshProUGUI relevanceScoreText;
    [SerializeField] public float relevanceScore;
    private RectTransform containerRectTransform;
    private RectTransform imageRectTransform;
    private AspectRatioFitter aspectRatioFitter;

    private void Awake()
    {
        containerRectTransform = GetComponent<RectTransform>();
        imageRectTransform = figureImage.GetComponent<RectTransform>();
        imageRectTransform.localScale = new Vector3(0.85f, 0.85f, 0.85f);
        
        // Add AspectRatioFitter if it doesn't exist
        aspectRatioFitter = figureImage.GetComponent<AspectRatioFitter>();
        if (aspectRatioFitter == null)
        {
            aspectRatioFitter = figureImage.gameObject.AddComponent<AspectRatioFitter>();
        }
        aspectRatioFitter.aspectMode = AspectRatioFitter.AspectMode.FitInParent;
    }

    public void SetContent(Texture2D texture, string title, float score)
    {
        if (texture != null)
        {
            figureImage.texture = texture;
            
            // Calculate the aspect ratio
            float aspectRatio = (float)texture.width / texture.height;
            aspectRatioFitter.aspectRatio = aspectRatio;
            
            // Calculate the required height
            float width = containerRectTransform.rect.width;
            float imageHeight = width / aspectRatio;
            
            // Set the container height to accommodate the image plus some padding
            float totalHeight = imageHeight + 20f; // 20 units padding
            containerRectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, totalHeight);
            
            titleText.text = $"{title}";
            relevanceScoreText.text = $"(Relevance Score: {score:F2})";
            relevanceScore = score;
        }
    }

    private void OnDestroy()
    {
        if (figureImage.texture != null)
        {
            Destroy(figureImage.texture);
        }
    }
}