document.getElementById("predictForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        age: parseInt(document.getElementById("age").value),
        gender: document.getElementById("gender").value,
        pack_years: parseFloat(document.getElementById("pack").value),
        radon_exposure: document.getElementById("radon").value,
        asbestos_exposure: document.getElementById("abs").value,
        secondhand_smoke_exposure: document.getElementById("smoke").value,
        copd_diagnosis: document.getElementById("copd").value,
        alcohol_consumption: document.getElementById("alcohol").value,
        family_history: document.getElementById("family").value
    };

    try {
        const response = await fetch("http://16.171.224.225:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        document.getElementById("result").innerText =
            `Prediction: ${result.prediction} `;

    } catch (error) {
        console.error("Error:", error);
    }
});