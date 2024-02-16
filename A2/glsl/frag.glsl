// Charles Gil, 260970950
#version 330 core

// Inputs (define in python code)
uniform vec3 lightPos; // World space
uniform vec3 objectColor; // Color used for ambient and diffused lighting
uniform bool compute;
in vec3 FragPos; // World space
in vec3 Normal;
in vec3 EyePos;

// Output
out vec4 f_color;

uniform float ambientIntensity = 0.3;

void main() {
	// Monkey head
	if(compute) {
		// Ambient lighting
		vec3 ambient = objectColor * ambientIntensity;

		// Lambertian lighting (diffuse)
		vec3 lightDir = normalize(lightPos - FragPos);
    	float diff = max(dot(Normal, lightDir), 0.0);
    	vec3 diffuse = diff * objectColor;

		// Blinn-Phone lighting (specular)
		vec3 viewDir = normalize(EyePos-FragPos);
    	vec3 halfwayDir = normalize(lightDir + viewDir); // Change view direction (eye position)
    	float spec = pow(max(dot(Normal, halfwayDir), 0.0), 100);
		vec3 specular = spec * vec3(1,1,1);

		// Combine
		vec3 result = ambient + diffuse + specular;
		f_color = vec4(result, 1.0);
	}

	// Labels
	else {
		f_color = vec4(objectColor, 1.0);
	}

}