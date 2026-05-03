$( document ).ready(function() {
    $("#get-stream").on("click", getStream);
    //barplot_new();
});

function getStream() {
    $("#output").empty()
    $("#output").append($("<div>", {"class": "start-char"}).text(">"));
    var query = $("#chat-input textarea").val();
    let current_chunk_idx = 0;

    var random_state = $("#random-seed").val();
    if (!random_state) {
        random_state = Math.floor(Math.random() * 10000) + 1
        console.log("RANDOM SATE USED: " + random_state)
    }

    if ($(".safenudge-toggle").is(':checked')) {
        safenudge = true
    } else {
        safenudge = false
    }


    var sleep_time = $("#sleep-time").val();

    // Create the data object to send in the request body
    const requestData = {
        init_prompt: query,
        safenudge: safenudge,
        k: 20,
        T: 1.3,
        max_new_tokens: 300,
        verbose: false,
        random_state: random_state,
        sleep_time: sleep_time
    };

    $.ajax({
        method: "POST",
        dataType: 'text',
        url: "/generate",  // Remove all query parameters
        data: requestData, // Send data in the body
        crossDomain: true,
        xhrFields: {
            onprogress: function (event) {
                let chunk = event.currentTarget.responseText; //.responseText;
                let chunks = chunk.split("}\n");
                //console.log(chunks.length);
                //console.log(chunks);
                for (let i = current_chunk_idx; i < chunks.length; i++) {
                    //console.log("--START--")
                    
                    let current_chunk = chunks[i];
                    
                    // Skip empty chunks
                    if (!current_chunk.trim()) {
                        continue;
                    }

                    if (current_chunk.slice(0,1) != "{") {
                        current_chunk = "{" + current_chunk;
                    }
                    if (current_chunk.slice(-1) != "}") {
                        current_chunk = current_chunk + "}";
                    }

                    try {
                        var obj = JSON.parse(current_chunk);
                        console.log(obj)
                        build(obj)
                    } catch (e) {
                        console.error("JSON parse error:", e);
                        console.error("Problematic chunk:", current_chunk);
                        console.error("Raw chunk:", chunks[i]);
                    }
                    
                    //console.log("--END--")
                }
                current_chunk_idx = chunks.length-1;
                //console.log(myArray)
                //let current_chunk = chunk.slice(current_chunk_idx);
                //console.log(current_chunk);
                //var obj = JSON.parse(current_chunk);
                //console.log(obj);
                //current_chunk_idx = current_chunk_idx + current_chunk.length;
                //console.log(event.currentTarget)
                
                //console.log("Received chunk:", chunk);
            }
        },
        success: function(data) {
            $("#data-store").attr('data-current-output', data);
        }
    })
}

async function build(obj) {
    texts = obj['texts'];
    probs = obj['probs'];

    if ($(".uncertainty-toggle").is(':checked')) {
        uncertainty = (1 - obj['probs'][obj['selected_idx']].toFixed(2));
    } else {
        uncertainty = 0
    }

    const data = texts.map((value, index) => ({ text: value, prob: probs[index].toFixed(2) }));

    $("#output").append($("<div>", {"class": "text", "data-obj": JSON.stringify(data), "data-idx-counter": obj['idx_counter']})
                .on("click", function() {
                    if ($(".safenudge-toggle").is(':checked')) {
                        alert('Probability viewing is not allowed while SafeNudge(TM) is activated.')
                    } else {
                        $(".selected").removeClass("selected")
                        $(this).addClass("selected")
                        d = JSON.parse($(this).attr('data-obj'));
                        barplot_new(d)
                    }
                })
                .text(obj['selected_text'])
                .css("background-color", "rgba(231, 76, 60," + uncertainty + ")")
    );

    console.log("HERE5", data['idx_counter'])
    
    //console.log("HERE4", $("#data-store").data('current_output'));
}

function isObject(obj)
{
    return obj !== undefined && obj !== null && obj.constructor == Object;
}

function handleData(data) {
    console.log("here3")
    console.log(data);
    //do some stuff
}

function barplot_new(data) {
    $('#barplot').empty()

    // Chart dimensions
    const margin = { top: 20, right: 30, bottom: 40, left: 100 };
    const width = 260 - margin.left - margin.right;
    const height = 415 - margin.top - margin.bottom; // You can add 15 pixels of extra height here... not sure why, maybe marigns/padding?

    // Create SVG container
    const svg = d3.select('#barplot')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`)
        .attr("fill", "white");

    // Create scales
    const xScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.prob)])
        .range([0, width]);

    const yScale = d3.scaleBand()
        .domain(data.map(d => d.text))
        .range([0, height])
        .padding(0.1);
        
    // Add X-axis
    svg.append('g')
        .attr('transform', `translate(0, ${height})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .attr("fill", "none");
        
    // Add Y-axis
    svg.append('g')
        .call(d3.axisLeft(yScale))
        .selectAll('text')
        .attr("fill", "#f5f6fa");
        
    // Draw bars
    svg.selectAll('.bar')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('y', d => yScale(d.text))
        .attr('x', 0)
        .attr('height', yScale.bandwidth())
        .attr('width', d => xScale(d.prob))
        .attr("fill", "white");

    // Add labels inside the bars
    svg.selectAll('.label')
        .data(data)
        .enter()
        .append('text')
        .attr('class', 'label')
        .attr('x', d => xScale(d.prob) - 5)
        .attr('y', d => yScale(d.text) + yScale.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('dx', '8px')
        .text(d => d.prob)
        .attr("fill", "white");

    $(".label").each(function() {
        console.log("here")
        var currentValue = $(this).text();
    
        // Check your condition here
        if (currentValue === "0.00") {
            $(this).text("<0.01");
        }
    });

    $(".tick").on( "click", function() {
        if ($(".safenudge-toggle").is(':checked')) {
            alert('Token editing is not allowed while SafeNudge(TM) is activated.')
        } else {
            var text = $(this).text()
            var idx_counter = $(".selected").attr('data-idx-counter');
    
            var query = $("#chat-input textarea").val();
            //alert(text + " " + idx_counter)
            console.log(idx_counter)
            //alert( text + "," + idx_counter );
            $("#output").empty()
            $("#output").append($("<div>", {"class": "start-char"}).text(">"));
            //var query = $("#chat-input textarea").val();
            let current_chunk_idx = 0;
        
            var random_state = $("#random-seed").val();
            if (!random_state) {
                random_state = Math.floor(Math.random() * 10000) + 1
                console.log("RANDOM SATE USED: " + random_state)
            }
        
            var sleep_time = $("#sleep-time").val();
            var current_output = $("#data-store").attr('data-current-output');

            // Create the data object to send in the request body
            const requestData = {
                init_prompt: query,
                content: current_output,
                token_pos: idx_counter,
                new_token: text,
                k: 20,
                T: 1.3,
                max_new_tokens: 300,
                sleep_time: sleep_time,
                verbose: true,
                random_state: random_state
            };

            $.ajax({
                method: "POST",
                dataType: 'text',
                url: "/regenerate",  // Remove all query parameters
                data: requestData,   // Send data in the body
                crossDomain: true,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                xhrFields: {
                    withCredentials: false,
                    onprogress: function (event) {
                        let chunk = event.currentTarget.responseText; //.responseText;
                        let chunks = chunk.split("}\n");
                        //console.log(chunks.length);
                        //console.log(chunks);
                        for (let i = current_chunk_idx; i < chunks.length; i++) {
                            //console.log("--START--")
                            
                            let current_chunk = chunks[i];
                            
                            // Skip empty chunks
                            if (!current_chunk.trim()) {
                                continue;
                            }
        
                            if (current_chunk.slice(0,1) != "{") {
                                current_chunk = "{" + current_chunk;
                            }
                            if (current_chunk.slice(-1) != "}") {
                                current_chunk = current_chunk + "}";
                            }
                            
                            try {
                                var obj = JSON.parse(current_chunk);
                                console.log(obj)
                                build(obj)
                            } catch (e) {
                                console.error("JSON parse error:", e);
                                console.error("Problematic chunk:", current_chunk);
                                console.error("Raw chunk:", chunks[i]);
                            }
                            
                            //console.log("--END--"))
                        }
                        current_chunk_idx = chunks.length-1;
                        //console.log(myArray)
                        //let current_chunk = chunk.slice(current_chunk_idx);
                        //console.log(current_chunk);
                        //var obj = JSON.parse(current_chunk);
                        //console.log(obj);
                        //current_chunk_idx = current_chunk_idx + current_chunk.length;
                        //console.log(event.currentTarget)
                        
                        //console.log("Received chunk:", chunk);
                    }
                },
                success: function(data) {
                    $("#data-store").attr('data-current-output', data);
                }
            });
        }


    });
}

// https://d3-graph-gallery.com/graph/barplot_horizontal.html
/*
function barplot(data) {
    // set the dimensions and margins of the graph
    var margin = {top: 20, right: 30, bottom: 40, left: 90},
    width = 280 - margin.left - margin.right,
    height = 580 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    var svg = d3.select("#barplot")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    // Parse the Data
    d3.csv("https://raw.githubusercontent.com/holtzy/data_to_viz/master/Example_dataset/7_OneCatOneNum_header.csv", function(data) {
        
        console.log(data)
        // Add X axis
        var x = d3.scaleLinear()
            .domain([0, 13000])
            .range([ 0, width]);

        svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))
            .selectAll("text")
            .attr("transform", "translate(-10,0)rotate(-45)")
            .style("text-anchor", "end");

        // Y axis
        var y = d3.scaleBand()
            .range([ 0, height ])
            .domain(data.map(function(d) { return d.Country; }))
            .padding(.1);

        svg.append("g")
            .call(d3.axisLeft(y))
            .attr("class", "y-axis")

        //Bars
        svg.selectAll("myRect")
            .data(data)
            .enter()
            .append("rect")
            .attr("x", x(0) )
            .attr("y", function(d) { return y(d.Country); })
            .attr("width", function(d) { return x(d.Value); })
            .attr("height", y.bandwidth() )
            .attr("fill", "#69b3a2")


            // .attr("x", function(d) { return x(d.Country); })
            // .attr("y", function(d) { return y(d.Value); })
            // .attr("width", x.bandwidth())
            // .attr("height", function(d) { return height - y(d.Value); })
            // .attr("fill", "#69b3a2")

        $(".y-axis .tick").on( "click", function() {
            // DEPRECATEDDD!!!!
            var text = $(this).text();
            var idx_counter = $(".selected").attr('idx-counter');
            console.log(idx_counter)
            alert( text, idx_counter );
        });

    })

}

function isNumber(value) {
    return typeof value === 'number' && isFinite(value);
}

function clickMe(){alert("I've been clicked!")}
*/