from integration.run_all import run_pipeline

def test_pipeline_runs():
    result = run_pipeline("assets/example.jpg")
    assert result["segments"] is not None
    assert result["past"] is not None
    assert result["future"] is not None
    assert "narrative" in result
    print("✅ Entegrasyon testi geçti.")

