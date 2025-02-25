$(document).ready(function() {
    let perPage = 10;
    let items = $(".result-item");
    let totalPages = Math.ceil(items.length / perPage);
    let currentPage = 1;

    function showPage(page) {
        items.hide();
        items.slice((page - 1) * perPage, page * perPage).show();
    }

    $(".next-page").click(function(e) {
        e.preventDefault();
        if (currentPage < totalPages) {
            currentPage++;
            showPage(currentPage);
        }
    });

    $(".prev-page").click(function(e) {
        e.preventDefault();
        if (currentPage > 1) {
            currentPage--;
            showPage(currentPage);
        }
    });

    showPage(currentPage);

    $("#search").on("keyup", function() {
        let value = $(this).val().toLowerCase();
        $(".result-item").each(function() {
            let text = $(this).text().toLowerCase();
            $(this).toggle(text.includes(value));
        });
    });
});