async function summarize() {
    const input = document.getElementById("inputText").value;
    const output = document.getElementById("output");
    output.textContent = "Generating summary...";

    const response = await fetch("/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input })
    });

    const data = await response.json();
    output.textContent = data.summary;
}
