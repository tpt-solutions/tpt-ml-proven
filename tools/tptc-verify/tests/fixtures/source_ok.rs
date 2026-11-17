// Fixture source: contains the runtime check for has_check only.
fn f(x: f64) -> Result<(), ()> {
    requires!(x >= 1.0, "fixture_linkage.has_check");
    Ok(())
}
